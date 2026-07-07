import type { GenStats } from "./multimodal.js";
import type { RuntimeStats } from "./inference.js";
import type { RunRecord } from "./history.js";
import { getModelPreset } from "./models.js";
import { vendorForModel } from "./history.js";

/** Build a RuntimeStats record from multimodal generation telemetry (tokens + first-token time). */
export function buildMultimodalStats(gen: GenStats, modelId: string): RuntimeStats {
  const elapsedMs = Math.max(0, gen.endedMs - gen.startedMs);
  const completionTokens = gen.completionTokens > 0 ? gen.completionTokens : undefined;
  const ttftS = gen.firstTokenMs !== undefined ? Math.max(0, (gen.firstTokenMs - gen.startedMs) / 1000) : undefined;
  const decodeSeconds = gen.firstTokenMs !== undefined ? Math.max(0.001, (gen.endedMs - gen.firstTokenMs) / 1000) : elapsedMs / 1000;
  const measured = completionTokens !== undefined && elapsedMs > 0 ? completionTokens / (elapsedMs / 1000) : undefined;
  return {
    rawText: "",
    model: modelId,
    backend: "transformers-js",
    finishReason: "stop",
    measuredElapsedMs: elapsedMs,
    completionTokens,
    totalTokens: completionTokens,
    measuredCompletionTokensPerSecond: measured,
    extra: {
      e2e_latency_s: elapsedMs / 1000,
      decode_tokens_per_s: completionTokens !== undefined ? completionTokens / decodeSeconds : undefined,
      time_to_first_token_s: ttftS,
    },
  };
}

/**
 * Map a generation's RuntimeStats to a persistable RunRecord, attributing the run to the model that
 * produced the stats (not whatever is currently selected). Returns null for stats without a model
 * (error placeholders) so failed runs never pollute history.
 */
export function runRecordFromStats(caseKey: string, stats: RuntimeStats): RunRecord | null {
  if (!stats.model || stats.model === "n/a") return null;
  const preset = getModelPreset(stats.model);
  const extra = stats.extra;
  return {
    ts: Date.now(),
    model: preset.shortName,
    vendor: vendorForModel(preset),
    backend: stats.backend === "transformers-js" ? "transformers-js" : "webllm",
    case: caseKey,
    promptTokens: stats.promptTokens ?? 0,
    completionTokens: stats.completionTokens ?? 0,
    decode: extra?.decode_tokens_per_s ?? stats.measuredCompletionTokensPerSecond ?? 0,
    ttft: extra?.time_to_first_token_s ?? 0,
    totalMs: stats.measuredElapsedMs,
    measured: stats.measuredCompletionTokensPerSecond ?? 0,
  };
}

/**
 * Word-count proxy for non-generative transcription runs: ASR pipelines (Whisper/Moonshine) produce
 * no token telemetry, so callers may log `{ completionTokens: wordCount }` instead. Exposed as an
 * explicit helper so the unit substitution is a visible policy, not a silent one.
 */
export function wordCountGenStats(text: string, startedMs: number, endedMs: number): GenStats {
  const words = text ? text.trim().split(/\s+/).filter(Boolean).length : 0;
  return { completionTokens: words, startedMs, endedMs };
}
