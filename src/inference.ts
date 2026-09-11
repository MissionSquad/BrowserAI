import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import type { ModelBackend, ModelPreset } from "./models.js";
import type { RuntimeParameters } from "./generation.js";
import { buildResponseFormat } from "./generation.js";
import { formatError } from "./errors.js";

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type LatencyBreakdown = Record<string, number[]>;

export type WebLLMUsageExtra = {
  e2e_latency_s?: number;
  prefill_tokens_per_s?: number;
  decode_tokens_per_s?: number;
  time_to_first_token_s?: number;
  time_per_output_token_s?: number;
  grammar_init_s?: number;
  grammar_per_token_s?: number;
  latencyBreakdown?: LatencyBreakdown;
  token_count_source?: string;
  [key: string]: unknown;
};

export type RuntimeStats = {
  rawText: string;
  model: string;
  backend: ModelBackend | "unknown";
  finishReason: string;
  measuredElapsedMs: number;
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  measuredCompletionTokensPerSecond?: number;
  extra?: WebLLMUsageExtra;
  latencyBreakdown?: LatencyBreakdown;
  legacyStats?: string;
  error?: string;
};

/* ------------------------------------------------------------------ */
/* Transformers.js structural types (the package ships loose typings). */
/* ------------------------------------------------------------------ */

export type TransformersTokenizer = {
  apply_chat_template?: (messages: unknown, options?: Record<string, unknown>) => unknown;
  encode?: (text: string, options?: Record<string, unknown>) => unknown;
  decode?: (tokens: unknown, options?: Record<string, unknown>) => string;
  batch_decode?: (tokens: unknown, options?: Record<string, unknown>) => string[];
  all_special_ids?: unknown[];
  (input: unknown, options?: Record<string, unknown>): unknown;
};

export type TransformersTextGenerationPipeline = ((input: unknown, options?: Record<string, unknown>) => Promise<unknown>) & {
  tokenizer?: TransformersTokenizer;
  model?: { dispose?: () => unknown };
  dispose?: () => unknown;
};

export type TransformersModule = {
  env?: Record<string, unknown>;
  pipeline: (task: string, model: string, options?: Record<string, unknown>) => Promise<TransformersTextGenerationPipeline>;
  TextStreamer?: new (tokenizer: TransformersTokenizer, options?: Record<string, unknown>) => unknown;
};

type TransformersGenerationTelemetry = {
  promptTokens?: number;
  completionTokens?: number;
  firstTokenOffsetMs?: number;
  tokenCallbackIntervalSeconds: number[];
  tokenCallbackCount: number;
  tokenCountSource: "streamer" | "tokenizer" | "unavailable";
};

/**
 * Live engine references owned by the client and passed into the stateless generation helpers.
 */
export type EngineHandles = {
  engine: MLCEngineInterface | null;
  transformersGenerator: TransformersTextGenerationPipeline | null;
  transformersTokenizer: TransformersTokenizer | null;
  transformersModule: TransformersModule | null;
  backend: ModelBackend | null;
  modelId: string | null;
};

export type GenerationResult = {
  text: string;
  stats: RuntimeStats;
};

export type GenerateOptions = {
  /** Structured cases pass a schema + json mode for WebLLM constrained decoding. */
  schema?: unknown;
  /** Stream tokens to this callback (receives the full accumulated text so far). */
  onDelta?: (fullText: string) => void;
  /** Reset the WebLLM KV cache before generating. Default true (stateless). Set false for multi-turn chat that relies on passed history — history is still passed explicitly, so reset stays safe. */
  resetChat?: boolean;
};

/** Sentinel substituted for blank model output; exported so callers can detect/skip it (e.g. TTS). */
export const EMPTY_RESPONSE = "[empty response]";

/**
 * Run one generation against whichever backend is loaded and return the text plus normalized stats.
 * Dispatches to WebLLM (streaming or blocking) or Transformers.js. Throws if no engine is loaded.
 */
export async function generate(
  handles: EngineHandles,
  messages: ChatMessage[],
  runtime: RuntimeParameters,
  preset: ModelPreset,
  options: GenerateOptions = {},
): Promise<GenerationResult> {
  const started = performance.now();

  if (handles.backend === "transformers-js") {
    if (!handles.transformersGenerator) throw new Error("Transformers.js pipeline is not loaded.");
    const telemetry = await runTransformers(handles, messages, runtime, preset, options.onDelta, started);
    const ended = performance.now();
    const text = telemetry.text || EMPTY_RESPONSE;
    const stats = collectTransformersStats(preset.id, runtime, telemetry, started, ended);
    return { text, stats };
  }

  if (!handles.engine) throw new Error("WebLLM engine is not loaded.");

  if (options.resetChat !== false) {
    await handles.engine.resetChat(false);
  }

  const responseFormat = options.schema !== undefined ? buildResponseFormat(runtime.jsonMode, options.schema) : undefined;
  const extraBody = buildExtraBody(runtime, preset);

  const outcome = options.onDelta
    ? await runWebLLMStream(handles.engine, messages, runtime, responseFormat, extraBody, options.onDelta)
    : await runWebLLMBlocking(handles.engine, messages, runtime, responseFormat, extraBody);

  const ended = performance.now();
  const stats = await collectWebLLMStats(handles, outcome.usage, outcome.finishReason, started, ended, outcome.firstTokenMs);
  return { text: outcome.text || EMPTY_RESPONSE, stats };
}

function buildExtraBody(runtime: RuntimeParameters, preset: ModelPreset): Record<string, unknown> | undefined {
  const extraBody: Record<string, unknown> = {};
  if (preset.disableThinkingSupported) {
    extraBody.enable_thinking = runtime.disableThinking === false;
  }
  if (runtime.latencyBreakdown) {
    extraBody.enable_latency_breakdown = true;
  }
  return Object.keys(extraBody).length > 0 ? extraBody : undefined;
}

type WebLLMOutcome = {
  text: string;
  finishReason: string;
  usage?: WebLLMUsage;
  firstTokenMs?: number;
};

type WebLLMUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  extra?: WebLLMUsageExtra;
};

async function runWebLLMBlocking(
  engine: MLCEngineInterface,
  messages: ChatMessage[],
  runtime: RuntimeParameters,
  responseFormat: ReturnType<typeof buildResponseFormat>,
  extraBody: Record<string, unknown> | undefined,
): Promise<WebLLMOutcome> {
  const response = await engine.chat.completions.create({
    messages,
    stream: false,
    n: 1,
    temperature: runtime.temperature,
    top_p: runtime.topP,
    max_tokens: runtime.maxTokens,
    repetition_penalty: runtime.repetitionPenalty,
    frequency_penalty: runtime.frequencyPenalty,
    presence_penalty: runtime.presencePenalty,
    seed: runtime.seed ?? undefined,
    response_format: responseFormat,
    extra_body: extraBody,
  });

  return {
    text: response.choices[0]?.message?.content ?? "",
    finishReason: response.choices[0]?.finish_reason ?? "unknown",
    usage: response.usage as WebLLMUsage | undefined,
  };
}

async function runWebLLMStream(
  engine: MLCEngineInterface,
  messages: ChatMessage[],
  runtime: RuntimeParameters,
  responseFormat: ReturnType<typeof buildResponseFormat>,
  extraBody: Record<string, unknown> | undefined,
  onDelta: (fullText: string) => void,
): Promise<WebLLMOutcome> {
  const stream = await engine.chat.completions.create({
    messages,
    stream: true,
    stream_options: { include_usage: true },
    temperature: runtime.temperature,
    top_p: runtime.topP,
    max_tokens: runtime.maxTokens,
    repetition_penalty: runtime.repetitionPenalty,
    frequency_penalty: runtime.frequencyPenalty,
    presence_penalty: runtime.presencePenalty,
    seed: runtime.seed ?? undefined,
    response_format: responseFormat,
    extra_body: extraBody,
  });

  let text = "";
  let finishReason = "unknown";
  let usage: WebLLMUsage | undefined;
  let firstTokenMs: number | undefined;

  for await (const chunk of stream) {
    const choice = chunk.choices[0];
    const delta = choice?.delta?.content ?? "";
    if (delta) {
      if (firstTokenMs === undefined) firstTokenMs = performance.now();
      text += delta;
      onDelta(text);
    }
    if (choice?.finish_reason) finishReason = choice.finish_reason;
    if (chunk.usage) usage = chunk.usage as WebLLMUsage;
  }

  return { text, finishReason, usage, firstTokenMs };
}

async function collectWebLLMStats(
  handles: EngineHandles,
  usage: WebLLMUsage | undefined,
  finishReason: string,
  started: number,
  ended: number,
  firstTokenMs?: number,
): Promise<RuntimeStats> {
  const elapsedMs = ended - started;
  const completionTokens = usage?.completion_tokens;
  const measured = typeof completionTokens === "number" && elapsedMs > 0 ? completionTokens / (elapsedMs / 1000) : undefined;
  // Prefer the engine's telemetry. When it is absent (some engine/model combos omit usage.extra),
  // synthesize the headline metrics from what we measured so they never blank out.
  const extra: WebLLMUsageExtra | undefined = usage?.extra
    ? { ...usage.extra, latencyBreakdown: normalizeLatencyBreakdown(usage.extra.latencyBreakdown) }
    : measured !== undefined || firstTokenMs !== undefined
      ? {
          decode_tokens_per_s: measured,
          e2e_latency_s: elapsedMs / 1000,
          time_to_first_token_s: firstTokenMs !== undefined ? Math.max(0, (firstTokenMs - started) / 1000) : undefined,
        }
      : undefined;

  let legacyStats: string | undefined;
  if (handles.engine && handles.modelId) {
    try {
      const engineStats = await handles.engine.runtimeStatsText(handles.modelId);
      legacyStats = engineStats.trim() || undefined;
    } catch (error) {
      legacyStats = `unavailable: ${formatError(error)}`;
    }
  }

  const stats: RuntimeStats = {
    rawText: "",
    model: handles.modelId ?? "n/a",
    backend: "webllm",
    finishReason,
    measuredElapsedMs: elapsedMs,
    promptTokens: usage?.prompt_tokens,
    completionTokens: usage?.completion_tokens,
    totalTokens: usage?.total_tokens,
    measuredCompletionTokensPerSecond: measured,
    extra,
    latencyBreakdown: extra?.latencyBreakdown,
    legacyStats,
  };
  stats.rawText = buildRuntimeStatsText(stats);
  return stats;
}

/* ------------------------------------------------------------------ */
/* Transformers.js path                                                */
/* ------------------------------------------------------------------ */

type TransformersRunResult = TransformersGenerationTelemetry & { text: string };

async function runTransformers(
  handles: EngineHandles,
  messages: ChatMessage[],
  runtime: RuntimeParameters,
  preset: ModelPreset,
  onDelta: ((fullText: string) => void) | undefined,
  // Origin clock shared with generate()'s `started` so TTFT and e2e latency are measured consistently.
  startedMs: number,
): Promise<TransformersRunResult> {
  const generator = handles.transformersGenerator!;
  const tokenizer = handles.transformersTokenizer ?? generator.tokenizer ?? null;
  const chatTemplateOptions = preset.disableThinkingSupported ? { enable_thinking: !runtime.disableThinking } : undefined;

  const telemetry: TransformersGenerationTelemetry = {
    promptTokens: countPromptTokens(tokenizer, messages, chatTemplateOptions),
    tokenCallbackIntervalSeconds: [],
    tokenCallbackCount: 0,
    tokenCountSource: "unavailable",
  };
  const tokenTimesMs: number[] = [];

  const generationOptions: Record<string, unknown> = {
    max_new_tokens: runtime.maxTokens,
    do_sample: runtime.temperature > 0,
    top_p: runtime.topP,
    repetition_penalty: runtime.repetitionPenalty,
    return_full_text: false,
  };
  if (runtime.temperature > 0) generationOptions.temperature = runtime.temperature;
  if (runtime.seed !== null) generationOptions.seed = runtime.seed;
  if (chatTemplateOptions) {
    generationOptions.tokenizer_encode_kwargs = chatTemplateOptions;
  }

  const Streamer = handles.transformersModule?.TextStreamer;
  let streamedText = "";
  if (tokenizer && Streamer) {
    generationOptions.streamer = new Streamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk: unknown) => {
        if (typeof chunk === "string" && onDelta) {
          streamedText += chunk;
          onDelta(streamedText);
        }
      },
      token_callback_function: (tokens: unknown) => {
        const now = performance.now();
        const count = Math.max(1, countTokenItems(tokens));
        for (let index = 0; index < count; index += 1) tokenTimesMs.push(now);
      },
    });
  }

  const output = await generator(messages, generationOptions);
  const text = extractGeneratedText(output) || EMPTY_RESPONSE;

  if (tokenTimesMs.length > 0) {
    telemetry.tokenCallbackCount = tokenTimesMs.length;
    telemetry.tokenCountSource = "streamer";
    telemetry.completionTokens = tokenTimesMs.length;
    telemetry.firstTokenOffsetMs = Math.max(0, tokenTimesMs[0] - startedMs);
    telemetry.tokenCallbackIntervalSeconds = tokenTimesMs.slice(1).map((time, index) => Math.max(0, (time - tokenTimesMs[index]) / 1000));
  } else {
    const fallback = countCompletionTokens(tokenizer, text);
    if (typeof fallback === "number") {
      telemetry.completionTokens = fallback;
      telemetry.tokenCountSource = "tokenizer";
    }
  }

  return { ...telemetry, text };
}

function collectTransformersStats(
  model: string,
  runtime: RuntimeParameters,
  telemetry: TransformersRunResult,
  started: number,
  ended: number,
): RuntimeStats {
  const elapsedMs = ended - started;
  const completionTokens = telemetry.completionTokens;
  const measured = typeof completionTokens === "number" && elapsedMs > 0 ? completionTokens / (elapsedMs / 1000) : undefined;
  const decodeSeconds = typeof telemetry.firstTokenOffsetMs === "number" ? Math.max(0, (elapsedMs - telemetry.firstTokenOffsetMs) / 1000) : undefined;
  const decode = typeof completionTokens === "number" && typeof decodeSeconds === "number" && decodeSeconds > 0 ? completionTokens / decodeSeconds : measured;
  const prefill =
    typeof telemetry.promptTokens === "number" && typeof telemetry.firstTokenOffsetMs === "number" && telemetry.firstTokenOffsetMs > 0
      ? telemetry.promptTokens / (telemetry.firstTokenOffsetMs / 1000)
      : undefined;
  const tpot =
    telemetry.tokenCallbackIntervalSeconds.length > 0
      ? telemetry.tokenCallbackIntervalSeconds.reduce((sum, value) => sum + value, 0) / telemetry.tokenCallbackIntervalSeconds.length
      : typeof completionTokens === "number" && completionTokens > 0
        ? elapsedMs / 1000 / completionTokens
        : undefined;
  const totalTokens =
    typeof telemetry.promptTokens === "number" && typeof completionTokens === "number" ? telemetry.promptTokens + completionTokens : undefined;
  const finishReason = typeof completionTokens === "number" && completionTokens >= runtime.maxTokens ? "length (estimated)" : "stop/unknown";
  const latencyBreakdown =
    telemetry.tokenCallbackIntervalSeconds.length > 0 ? { transformersTokenIntervalTime: telemetry.tokenCallbackIntervalSeconds } : undefined;

  const stats: RuntimeStats = {
    rawText: "",
    model,
    backend: "transformers-js",
    finishReason,
    measuredElapsedMs: elapsedMs,
    promptTokens: telemetry.promptTokens,
    completionTokens,
    totalTokens,
    measuredCompletionTokensPerSecond: measured,
    extra: {
      e2e_latency_s: elapsedMs / 1000,
      prefill_tokens_per_s: prefill,
      decode_tokens_per_s: decode,
      time_to_first_token_s: typeof telemetry.firstTokenOffsetMs === "number" ? telemetry.firstTokenOffsetMs / 1000 : undefined,
      time_per_output_token_s: tpot,
      latencyBreakdown,
      token_count_source: telemetry.tokenCountSource,
    },
    latencyBreakdown,
  };
  stats.rawText = buildRuntimeStatsText(stats);
  return stats;
}

export function collectGenericStats(model: string, backend: ModelBackend | "unknown", started: number, ended: number, error?: string): RuntimeStats {
  const stats: RuntimeStats = {
    rawText: "",
    model,
    backend,
    finishReason: "error",
    measuredElapsedMs: ended - started,
    error,
  };
  stats.rawText = buildRuntimeStatsText(stats);
  return stats;
}

/* ------------------------------------------------------------------ */
/* Shared helpers                                                      */
/* ------------------------------------------------------------------ */

export function buildRuntimeStatsText(stats: RuntimeStats): string {
  const backendLabel = stats.backend === "transformers-js" ? "Transformers.js" : stats.backend === "webllm" ? "WebLLM" : "unknown";
  const lines = [
    `model: ${stats.model}`,
    `backend: ${backendLabel}`,
    `finish_reason: ${stats.finishReason}`,
    `measured_elapsed_ms: ${stats.measuredElapsedMs.toFixed(0)}`,
    "",
    "Token usage:",
    `  prompt_tokens: ${stats.promptTokens ?? "n/a"}`,
    `  completion_tokens: ${stats.completionTokens ?? "n/a"}`,
    `  total_tokens: ${stats.totalTokens ?? "n/a"}`,
    `  measured_completion_tokens_per_second: ${stats.measuredCompletionTokensPerSecond?.toFixed(2) ?? "n/a"}`,
  ];

  if (stats.extra) {
    lines.push(
      "",
      stats.backend === "transformers-js" ? "Transformers.js measured telemetry:" : "WebLLM usage.extra:",
      `  e2e_latency_s: ${formatMaybe(stats.extra.e2e_latency_s, 3)}`,
      `  prefill_tokens_per_s: ${formatMaybe(stats.extra.prefill_tokens_per_s, 2)}`,
      `  decode_tokens_per_s: ${formatMaybe(stats.extra.decode_tokens_per_s, 2)}`,
      `  time_to_first_token_s: ${formatMaybe(stats.extra.time_to_first_token_s, 3)}`,
      `  time_per_output_token_s: ${formatMaybe(stats.extra.time_per_output_token_s, 4)}`,
    );
    if (typeof stats.extra.grammar_init_s === "number") lines.push(`  grammar_init_s: ${stats.extra.grammar_init_s.toFixed(3)}`);
    if (typeof stats.extra.grammar_per_token_s === "number") lines.push(`  grammar_per_token_s: ${stats.extra.grammar_per_token_s.toFixed(6)}`);
    if (stats.backend === "transformers-js" && typeof stats.extra.token_count_source === "string") {
      lines.push(`  token_count_source: ${stats.extra.token_count_source}`);
    }
    if (stats.latencyBreakdown && Object.keys(stats.latencyBreakdown).length > 0) {
      lines.push("", "Latency breakdown:", JSON.stringify(stats.latencyBreakdown, null, 2));
    }
  } else {
    lines.push("", `${backendLabel} backend telemetry: n/a`);
  }

  if (stats.error) lines.push("", `error: ${stats.error}`);
  if (stats.legacyStats) lines.push("", "Legacy WebLLM runtimeStatsText:", stats.legacyStats);

  return lines.join("\n");
}

export function normalizeLatencyBreakdown(value: unknown): LatencyBreakdown | undefined {
  if (!value || typeof value !== "object") return undefined;
  const result: LatencyBreakdown = {};
  for (const [key, child] of Object.entries(value)) {
    if (Array.isArray(child)) {
      const numbers = child.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
      if (numbers.length > 0) result[key] = numbers;
    }
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function extractGeneratedText(output: unknown): string {
  if (typeof output === "string") return output;
  if (Array.isArray(output)) return output.map(extractGeneratedText).filter(Boolean).join("\n");
  if (!output || typeof output !== "object") return "";

  const record = output as Record<string, unknown>;
  const generated = record.generated_text;
  if (typeof generated === "string") return generated;
  if (Array.isArray(generated)) {
    const assistant = generated
      .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
      .filter((item) => item.role === "assistant" && typeof item.content === "string");
    const last = assistant.at(-1);
    if (typeof last?.content === "string") return last.content;
    return generated.map(extractGeneratedText).filter(Boolean).join("\n");
  }
  for (const key of ["text", "content", "answer"] as const) {
    if (typeof record[key] === "string") return record[key] as string;
  }
  return JSON.stringify(output, null, 2);
}

function countPromptTokens(
  tokenizer: TransformersTokenizer | null,
  messages: ChatMessage[],
  chatTemplateOptions?: Record<string, unknown>,
): number | undefined {
  if (!tokenizer) return undefined;
  try {
    if (typeof tokenizer.apply_chat_template === "function") {
      // Count the same prompt the pipeline generates, including MiniCPM5's optional empty think block.
      const tokenized = tokenizer.apply_chat_template(messages, { tokenize: true, add_generation_prompt: true, ...chatTemplateOptions });
      const count = extractTokenCount(tokenized);
      if (typeof count === "number") return count;
    }
  } catch {
    // Fall through to plain-text tokenization.
  }
  try {
    const prompt = messages.map((message) => `${message.role}: ${message.content}`).join("\n\n") + "\n\nassistant:";
    if (typeof tokenizer.encode === "function") {
      const count = extractTokenCount(tokenizer.encode(prompt, { add_special_tokens: true }));
      if (typeof count === "number") return count;
    }
    return extractTokenCount(tokenizer(prompt, { add_special_tokens: true }));
  } catch {
    return undefined;
  }
}

function countCompletionTokens(tokenizer: TransformersTokenizer | null, text: string): number | undefined {
  if (!tokenizer || !text || text === EMPTY_RESPONSE) return undefined;
  try {
    if (typeof tokenizer.encode === "function") return extractTokenCount(tokenizer.encode(text, { add_special_tokens: false }));
    return extractTokenCount(tokenizer(text, { add_special_tokens: false }));
  } catch {
    return undefined;
  }
}

function countTokenItems(value: unknown): number {
  return extractTokenCount(value) ?? 1;
}

function extractTokenCount(value: unknown): number | undefined {
  if (value == null) return undefined;
  if (typeof value === "number" || typeof value === "bigint") return 1;
  if (ArrayBuffer.isView(value)) return (value as ArrayBufferView & { length?: number }).length;
  if (Array.isArray(value)) {
    if (value.length === 0) return 0;
    if (value.every((item) => typeof item === "number" || typeof item === "bigint")) return value.length;
    const childCounts = value.map(extractTokenCount).filter((count): count is number => typeof count === "number");
    return childCounts.length > 0 ? childCounts.reduce((sum, count) => sum + count, 0) : value.length;
  }
  if (typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  for (const key of ["input_ids", "data", "tokens", "ids"] as const) {
    const count = extractTokenCount(record[key]);
    if (typeof count === "number") return count;
  }
  const dims = record.dims;
  if (Array.isArray(dims) && dims.every((dimension) => typeof dimension === "number")) {
    return dims.reduce((product, dimension) => product * dimension, 1);
  }
  return undefined;
}

function formatMaybe(value: unknown, digits: number): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}
