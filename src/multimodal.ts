import { type ModelPreset, type MultimodalRuntime, SUPERTONIC_2_MODEL_ID } from "./models.js";
import { AUDIO_SR } from "./audio/pcm.js";

/** The kokoro-js CDN bundle loaded by default (kokoro-js ships its own Transformers.js runtime). */
export const DEFAULT_KOKORO_CDN = "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm";

/**
 * In-browser multimodal inference (STT / TTS / VLM / Gemma image+audio) for Transformers.js models.
 * A faithful port of small-ai's verified inference recipes. The Transformers.js package ships broad
 * runtime exports with loose types, so the dynamic surface is modelled structurally here (an
 * isolated, justified use of `any` for a truly untyped external SDK).
 */

/* ------------------------------------------------------------------ */
/* Loose structural typings for the Transformers.js exports we use     */
/* ------------------------------------------------------------------ */

type AnyRecord = Record<string, unknown>;
type FromPretrained = { from_pretrained: (id: string, opts?: AnyRecord) => Promise<any> };
type PipelineFn = (task: string, model: string, opts?: AnyRecord) => Promise<any>;

/** The subset of Transformers.js runtime exports the multimodal paths rely on. */
export type MultimodalTransformers = {
  env?: AnyRecord;
  pipeline: PipelineFn;
  AutoProcessor: FromPretrained;
  AutoModelForVision2Seq: FromPretrained;
  AutoModelForImageTextToText: FromPretrained;
  AutoModelForAudioTextToText: FromPretrained;
  Gemma4ForConditionalGeneration: FromPretrained;
  TextStreamer: new (tokenizer: unknown, opts?: AnyRecord) => unknown;
  load_image: (src: string) => Promise<any>;
  read_audio: (url: string, samplingRate: number) => Promise<Float32Array>;
};

/** Transformers.js RawAudio result (verified shape from the pipeline/kokoro output). */
export type RawAudio = { audio: Float32Array; sampling_rate: number; toBlob: () => Blob };

/** A text sink for streaming TTS: push text incrementally, then close(). (kokoro-js TextSplitterStream.) */
export type TtsTextSplitter = { push: (...texts: string[]) => void; flush: () => void; close: () => void };

/**
 * The kokoro-js KokoroTTS surface the SDK uses — whole-text generate plus sentence-streaming stream.
 * Verified against kokoro-js@1.2.1 types/kokoro.d.ts: stream(text: string | TextSplitterStream,
 * { voice, speed, split_pattern }?) → AsyncGenerator<{ text; phonemes; audio: RawAudio }, void, void>.
 */
type KokoroInstance = {
  generate: (text: string, opts: AnyRecord) => Promise<RawAudio>;
  stream: (input: TtsTextSplitter | string, opts: AnyRecord) => AsyncIterable<{ text: string; phonemes: string; audio: RawAudio }>;
};

/** The module shape a kokoro loader must resolve to (what the kokoro-js ESM bundle exports). */
export type KokoroModule = {
  KokoroTTS: { from_pretrained: (id: string, opts?: Record<string, unknown>) => Promise<unknown> };
  TextSplitterStream: new () => TtsTextSplitter;
};

const MAX_PIXELS = 1_048_576; // ~1.05 MP — larger overflows the q4f16 Qwen-family vision encoder

/**
 * Speaker-embedding URL for a Supertonic voice id (F1–F5 / M1–M5), resolved against the loaded model's
 * own repo so Supertonic v1 and v2 each fetch their own `voices/*.bin`. Defaults to F1 when unset.
 */
export function supertonicVoiceUrl(modelId: string, voiceId?: string): string {
  const id = voiceId && /^[FM][1-5]$/.test(voiceId) ? voiceId : "F1";
  return `https://huggingface.co/${modelId}/resolve/main/voices/${id}.bin`;
}

export type MultimodalDevice = "webgpu" | "wasm";
export type ProgressCallback = (info: { status?: string; progress?: number; file?: string }) => void;

/** Real generation telemetry captured from the streamer (token counts + first-token time). */
export type GenStats = { completionTokens: number; firstTokenMs?: number; startedMs: number; endedMs: number };
/** VLM / Gemma generation result: text plus telemetry. */
export type VlmResult = { text: string } & GenStats;
export type SttResult = { text: string; segments: Array<{ start: number; end: number; text: string }>; gen?: GenStats };

/** Count tokens from a TextStreamer token_callback payload (array / typed array / tensor). */
function countTokenItems(value: unknown): number {
  if (value == null) return 0;
  if (ArrayBuffer.isView(value)) return (value as ArrayBufferView & { length?: number }).length ?? 1;
  if (Array.isArray(value)) return value.length;
  const record = value as { dims?: number[]; data?: { length?: number }; size?: number };
  if (record?.data && typeof record.data.length === "number") return record.data.length;
  if (Array.isArray(record?.dims)) return record.dims.reduce((product, dim) => product * dim, 1);
  if (typeof record?.size === "number") return record.size;
  return 1;
}

/** A TextStreamer that accumulates text (onDelta) AND counts tokens + records first-token time. */
function makeCountingStreamer(
  Streamer: MultimodalTransformers["TextStreamer"],
  tokenizer: unknown,
  onDelta: ((text: string) => void) | undefined,
  sink: { text: string; completionTokens: number; firstTokenMs?: number },
): unknown {
  return new Streamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk: unknown) => {
      if (typeof chunk === "string") {
        if (sink.firstTokenMs === undefined) sink.firstTokenMs = performance.now();
        sink.text += chunk;
        onDelta?.(sink.text);
      }
    },
    token_callback_function: (tokens: unknown) => {
      sink.completionTokens += countTokenItems(tokens);
    },
  });
}

/** A loaded multimodal runtime, discriminated by kind. */
export type MultimodalHandle =
  | { kind: "stt"; transcriber: (input: unknown, opts?: AnyRecord) => Promise<any> }
  | { kind: "tts-kokoro"; tts: KokoroInstance; TextSplitterStream: new () => TtsTextSplitter }
  | { kind: "tts-pipeline"; synth: (text: string, opts: AnyRecord) => Promise<RawAudio>; modelId: string }
  | {
      kind: "vlm";
      processor: any;
      model: any;
      Streamer: MultimodalTransformers["TextStreamer"];
      loadImage: MultimodalTransformers["load_image"];
      /** >0 → downscale large images to this pixel budget (imagetext family only). */
      maxPixels: number;
      /** SmolVLM passes the image as [image] + a trailing {}; others pass a bare image. */
      imageArray: boolean;
      maxNewTokens: number;
    }
  | {
      kind: "gemma";
      processor: any;
      model: any;
      Streamer: MultimodalTransformers["TextStreamer"];
      loadImage: MultimodalTransformers["load_image"];
    }
  | {
      /**
       * Audio-text-to-text LLM ASR (Granite Speech, Ultravox, non-realtime Voxtral): the prompt carries
       * the model's audio placeholder token and the processor is called as processor(text, audio).
       */
      kind: "audiolm";
      processor: any;
      model: any;
      Streamer: MultimodalTransformers["TextStreamer"];
    }
  | {
      /**
       * Voxtral realtime streaming ASR: the processor takes audio ONLY (processor(audio) → input_features)
       * and generate({ input_features }) stops itself when the audio is exhausted — no text prompt.
       */
      kind: "audiolm-stream";
      processor: any;
      model: any;
      Streamer: MultimodalTransformers["TextStreamer"];
    };

/* ------------------------------------------------------------------ */
/* Loading                                                             */
/* ------------------------------------------------------------------ */

export type LoadMultimodalOptions = {
  /**
   * How to load the kokoro-js module for `tts-kokoro` presets. Defaults to a dynamic ESM import of
   * {@link DEFAULT_KOKORO_CDN} (browser-only; bundlers must leave the URL external). Hosts that
   * bundle kokoro-js themselves can pass `load: () => import("kokoro-js")`.
   */
  kokoro?: { url?: string; load?: () => Promise<KokoroModule> };
};

function vlmDtype(preset: ModelPreset, runtime: MultimodalRuntime): unknown {
  if (runtime === "vlm-vision2seq") return { embed_tokens: "fp16", vision_encoder: "q4", decoder_model_merged: "q4" };
  // imagetext family: GLM-OCR ships a single q4f16 dtype; Qwen3-VL uses a per-module mixed map.
  if (preset.transformersDtype) return preset.transformersDtype;
  return { embed_tokens: "fp16", vision_encoder: "fp16", decoder_model_merged: "q4f16" };
}

/**
 * Load the shared pieces of an audio-text-to-text model (Granite Speech, Voxtral realtime): the
 * AutoProcessor and the AutoModelForAudioTextToText model on WebGPU. All three ONNX modules
 * (audio_encoder, embed_tokens, decoder_model_merged) publish a q4f16 variant, so a flat dtype loads
 * them uniformly; preset.transformersDtype overrides it if a model needs a different mix. The caller
 * tags the returned bundle with the right handle `kind` (audiolm vs audiolm-stream).
 */
async function loadAudioLmSessions(
  preset: ModelPreset,
  transformers: MultimodalTransformers,
  progress_callback: ProgressCallback,
): Promise<{ processor: any; model: any; Streamer: MultimodalTransformers["TextStreamer"] }> {
  const processor = await transformers.AutoProcessor.from_pretrained(preset.id, { progress_callback });
  const dtype: string = preset.transformersDtype ?? "q4f16";
  const model = await transformers.AutoModelForAudioTextToText.from_pretrained(preset.id, { device: "webgpu", dtype, progress_callback });
  return { processor, model, Streamer: transformers.TextStreamer };
}

export async function loadMultimodal(
  preset: ModelPreset,
  transformers: MultimodalTransformers,
  device: MultimodalDevice,
  progress_callback: ProgressCallback,
  options: LoadMultimodalOptions = {},
): Promise<MultimodalHandle> {
  const runtime = preset.mmRuntime;
  if (!runtime) throw new Error(`Model ${preset.shortName} has no multimodal runtime.`);

  switch (runtime) {
    case "stt": {
      // Whisper/Moonshine load an fp32 encoder + a quantized decoder (encoder quantization hurts
      // accuracy). Some ASR models document a single flat dtype instead — e.g. Cohere's cohere_asr,
      // whose fp32 encoder is multiple GB, ships a q4 WebGPU recipe. Honor preset.transformersDtype
      // when set so those models are not force-loaded with an fp32 encoder.
      const dtype: string | Record<string, string> =
        preset.transformersDtype ??
        (device === "webgpu"
          ? { encoder_model: "fp32", decoder_model_merged: "q4" }
          : { encoder_model: "fp32", decoder_model_merged: "q8" });
      const transcriber = await transformers.pipeline("automatic-speech-recognition", preset.id, { device, dtype, progress_callback });
      return { kind: "stt", transcriber };
    }
    case "tts-pipeline": {
      const synth = await transformers.pipeline("text-to-speech", preset.id, { device, dtype: "fp32", progress_callback });
      return { kind: "tts-pipeline", synth, modelId: preset.id };
    }
    case "tts-kokoro": {
      // kokoro-js bundles its own Transformers.js runtime; by default load it from the CDN.
      const kokoro = await loadKokoroModule(options.kokoro);
      const tts = await kokoro.KokoroTTS.from_pretrained(preset.id, {
        dtype: device === "webgpu" ? "fp32" : "q8",
        device,
        progress_callback,
      });
      return { kind: "tts-kokoro", tts: tts as KokoroInstance, TextSplitterStream: kokoro.TextSplitterStream };
    }
    case "vlm-vision2seq":
    case "vlm-imagetext": {
      const processor = await transformers.AutoProcessor.from_pretrained(preset.id, { progress_callback });
      const ModelClass = runtime === "vlm-vision2seq" ? transformers.AutoModelForVision2Seq : transformers.AutoModelForImageTextToText;
      const model = await ModelClass.from_pretrained(preset.id, { device: "webgpu", dtype: vlmDtype(preset, runtime), progress_callback });
      return {
        kind: "vlm",
        processor,
        model,
        Streamer: transformers.TextStreamer,
        loadImage: transformers.load_image,
        maxPixels: runtime === "vlm-imagetext" ? MAX_PIXELS : 0,
        imageArray: runtime === "vlm-vision2seq",
        // Generous caps so descriptions/OCR are not truncated mid-output (any OCR model → 2048).
        maxNewTokens: preset.id.toLowerCase().includes("ocr") ? 2048 : 1024,
      };
    }
    case "gemma-mm": {
      const processor = await transformers.AutoProcessor.from_pretrained(preset.id, { progress_callback });
      const model = await transformers.Gemma4ForConditionalGeneration.from_pretrained(preset.id, { device: "webgpu", dtype: "q4f16", progress_callback });
      return { kind: "gemma", processor, model, Streamer: transformers.TextStreamer, loadImage: transformers.load_image };
    }
    case "audio-text-to-text":
      // Granite Speech / Ultravox / non-realtime Voxtral: audio placeholder token in the prompt.
      return { kind: "audiolm", ...(await loadAudioLmSessions(preset, transformers, progress_callback)) };
    case "voxtral-realtime":
      // Voxtral realtime: same load, but the processor is audio-only and generate is self-stopping.
      return { kind: "audiolm-stream", ...(await loadAudioLmSessions(preset, transformers, progress_callback)) };
    default:
      throw new Error(`Unsupported multimodal runtime: ${runtime as string}`);
  }
}

async function loadKokoroModule(options?: LoadMultimodalOptions["kokoro"]): Promise<KokoroModule> {
  if (options?.load) return options.load();
  const url = options?.url ?? DEFAULT_KOKORO_CDN;
  // Remote dynamic ESM import: browsers support it natively; bundlers must leave the URL external
  // (the indirection through a variable prevents most bundlers from trying to resolve it).
  return (await import(/* @vite-ignore */ /* webpackIgnore: true */ url)) as KokoroModule;
}

export async function disposeMultimodal(handle: MultimodalHandle | null): Promise<void> {
  if (!handle) return;
  const model = (handle as { model?: { dispose?: () => unknown } }).model;
  try {
    if (typeof model?.dispose === "function") await model.dispose();
  } catch {
    // Best-effort; the WebGPU context is released when the tab/model is replaced.
  }
}

/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */

/**
 * Gemma's audio feature-extractor only "hears" ~30s per generate() call, so a longer clip must be
 * split into ≤30s windows and transcribed window-by-window. Cap the total work at 12 minutes.
 */
const GEMMA_AUDIO_CHUNK_SEC = 30;
const GEMMA_AUDIO_MAX_SEC = 12 * 60; // 720s ceiling on chunked transcription
const GEMMA_TRANSCRIBE_PROMPT = "Transcribe this audio verbatim.";

/** A half-open [start, end) span of sample indices within a decoded 16 kHz audio buffer. */
export type AudioWindow = { start: number; end: number };

/**
 * Plan the consecutive, non-overlapping ≤30s sample windows a Gemma transcription splits a clip
 * into. The clip is first truncated to the 12-minute ceiling, then walked in
 * {@link GEMMA_AUDIO_CHUNK_SEC} windows (the final window may be shorter). Pure and exported so the
 * boundary math (truncation ceiling, last partial window) is unit-tested independently of a model.
 */
export function planGemmaAudioWindows(totalSamples: number): AudioWindow[] {
  const chunkSamples = GEMMA_AUDIO_CHUNK_SEC * AUDIO_SR;
  const maxSamples = GEMMA_AUDIO_MAX_SEC * AUDIO_SR;
  const usable = Math.min(Math.max(0, Math.floor(totalSamples)), maxSamples);
  const windows: AudioWindow[] = [];
  for (let start = 0; start < usable; start += chunkSamples) {
    windows.push({ start, end: Math.min(start + chunkSamples, usable) });
  }
  return windows;
}

/**
 * Transcribe a 16 kHz mono Float32Array. ASR-pipeline models (Whisper/Moonshine — handle kind
 * "stt") additionally accept an audio URL, which the Transformers.js pipeline fetches and decodes;
 * the generative STT recipes (Gemma, Granite Speech, Voxtral) require decoded PCM.
 * @throws {Error} when a URL is passed to a model that cannot fetch audio itself.
 */
export async function transcribe(handle: MultimodalHandle, audio: Float32Array | string, onDelta?: (text: string) => void): Promise<SttResult> {
  if (typeof audio === "string" && handle.kind !== "stt") {
    throw new Error("URL audio input is only supported by ASR-pipeline models (Whisper/Moonshine). Decode to 16 kHz mono PCM first (decodeAudioTo16kMono).");
  }
  if (handle.kind === "gemma") {
    // Callers supply decoded 16 kHz samples; chunk anything past Gemma's ~30s window.
    return transcribeGemmaChunked(handle, audio as Float32Array, onDelta);
  }
  if (handle.kind === "audiolm") {
    // Audio-LLM (Granite Speech, etc.): single-pass generate over the whole clip, no segments.
    return transcribeAudioLm(handle, audio as Float32Array, onDelta);
  }
  if (handle.kind === "audiolm-stream") {
    // Voxtral realtime: audio-only, self-stopping generate over the whole clip, no segments.
    return transcribeVoxtralRealtime(handle, audio as Float32Array, onDelta);
  }
  if (handle.kind !== "stt") throw new Error("Loaded model cannot transcribe audio.");
  // Whisper supports segment timestamps; Moonshine may not — try, then fall back to a plain pass.
  let out: any;
  try {
    out = await handle.transcriber(audio, { return_timestamps: true, chunk_length_s: 30, stride_length_s: 5 });
  } catch {
    out = await handle.transcriber(audio);
  }
  const text = String(out?.text ?? "").trim();
  const chunks: Array<{ timestamp?: [number, number]; text?: string }> = Array.isArray(out?.chunks) ? out.chunks : [];
  const segments = chunks
    .filter((c) => Array.isArray(c.timestamp))
    .map((c) => ({ start: c.timestamp![0] ?? 0, end: c.timestamp![1] ?? 0, text: String(c.text ?? "").trim() }))
    .filter((s) => s.text.length > 0);
  return { text, segments };
}

/**
 * Transcribe with Gemma, splitting audio longer than {@link GEMMA_AUDIO_CHUNK_SEC} into consecutive
 * non-overlapping windows (Gemma's feature-extractor only hears ~30s per pass). Clips beyond
 * {@link GEMMA_AUDIO_MAX_SEC} are truncated to that ceiling. Each window becomes a timestamped segment;
 * partial text streams across windows (already-finished windows are prefixed onto the live one), and
 * the returned telemetry sums token counts across all windows for accurate stats.
 */
async function transcribeGemmaChunked(
  handle: Extract<MultimodalHandle, { kind: "gemma" }>,
  audio: Float32Array,
  onDelta?: (text: string) => void,
): Promise<SttResult> {
  const windows = planGemmaAudioWindows(audio.length);

  // Fast path: a single ≤30s window (or empty audio) needs no chunk bookkeeping.
  if (windows.length <= 1) {
    const clip = windows[0] ? audio.subarray(windows[0].start, windows[0].end) : audio;
    const result = await runGemma(handle, { audio: clip, prompt: GEMMA_TRANSCRIBE_PROMPT }, onDelta);
    return { text: result.text, segments: [], gen: result };
  }

  const segments: Array<{ start: number; end: number; text: string }> = [];
  const parts: string[] = [];
  const startedMs = performance.now();
  let completionTokens = 0;
  let firstTokenMs: number | undefined;

  for (const window of windows) {
    const prior = parts.join(" ");
    const result = await runGemma(handle, { audio: audio.subarray(window.start, window.end), prompt: GEMMA_TRANSCRIBE_PROMPT }, (partial) => {
      onDelta?.(prior ? `${prior} ${partial}` : partial);
    });
    const windowText = result.text.trim();
    if (windowText) {
      segments.push({ start: window.start / AUDIO_SR, end: window.end / AUDIO_SR, text: windowText });
      parts.push(windowText);
      onDelta?.(parts.join(" "));
    }
    completionTokens += result.completionTokens;
    if (firstTokenMs === undefined && result.firstTokenMs !== undefined) firstTokenMs = result.firstTokenMs;
  }

  return { text: parts.join(" "), segments, gen: { completionTokens, firstTokenMs, startedMs, endedMs: performance.now() } };
}

const AUDIO_LM_TRANSCRIBE_PROMPT = "Transcribe this audio verbatim.";

/**
 * Transcribe with an audio-text-to-text LLM (Granite Speech / Voxtral). These models take a decoded
 * 16 kHz clip plus a text instruction and *generate* the transcript, so — unlike the Whisper pipeline
 * — there are no per-segment timestamps. The whole clip is run in a single pass and returned as plain
 * text plus real token/latency telemetry.
 */
async function transcribeAudioLm(
  handle: Extract<MultimodalHandle, { kind: "audiolm" }>,
  audio: Float32Array,
  onDelta?: (text: string) => void,
): Promise<SttResult> {
  const result = await runAudioLm(handle, audio, AUDIO_LM_TRANSCRIBE_PROMPT, onDelta);
  return { text: result.text, segments: [], gen: result };
}

/**
 * Generate text from a 16 kHz mono clip + a text prompt with an audio-text-to-text model. The prompt
 * must carry the model's audio placeholder token: the processor (verified GraniteSpeechProcessor code)
 * requires the token to be present in the text and replaces it with N audio tokens sized to the clip.
 * So the placeholder is embedded in the user turn's (string) content, rendered with
 * `apply_chat_template({ tokenize: false })`, then `processor(text, audio)` produces the model inputs
 * (this is why a list-content chat turn fails — the model's chat template drops the audio marker).
 * Streams partial text via onDelta; returns text + telemetry.
 */
export async function runAudioLm(
  handle: Extract<MultimodalHandle, { kind: "audiolm" }>,
  audio: Float32Array,
  prompt: string,
  onDelta?: (text: string) => void,
): Promise<VlmResult> {
  const audioToken: string = (handle.processor?.config?.audio_token as string | undefined) ?? "<|audio|>";
  const conversation = [{ role: "user", content: `${audioToken}${prompt}` }];
  const text = handle.processor.apply_chat_template(conversation, { add_generation_prompt: true, tokenize: false });
  const inputs = await handle.processor(text, audio);

  const sink = { text: "", completionTokens: 0, firstTokenMs: undefined as number | undefined };
  const startedMs = performance.now();
  const streamer = makeCountingStreamer(handle.Streamer, handle.processor.tokenizer, onDelta, sink);
  await handle.model.generate({ ...inputs, max_new_tokens: 1024, do_sample: false, streamer });
  return { text: sink.text.trim(), completionTokens: sink.completionTokens, firstTokenMs: sink.firstTokenMs, startedMs, endedMs: performance.now() };
}

/** Transcribe with Voxtral realtime (single-pass over the whole clip). No timestamped segments. */
async function transcribeVoxtralRealtime(
  handle: Extract<MultimodalHandle, { kind: "audiolm-stream" }>,
  audio: Float32Array,
  onDelta?: (text: string) => void,
): Promise<SttResult> {
  const result = await runVoxtralRealtime(handle, audio, onDelta);
  return { text: result.text, segments: [], gen: result };
}

/**
 * Transcribe a 16 kHz mono clip with a Voxtral realtime model. This is a genuine *streaming* model:
 * its `generate` consumes audio as it decodes through a causal streaming encoder, so it is NOT a
 * single-pass `generate({ input_features })` call. Faithful to the verified 4.2.0 recipe (from
 * `tests/models/voxtral_realtime/test_modeling_voxtral_realtime.js`):
 *   - Seed `input_ids` from the first streaming chunk (BOS + pad/delay tokens).
 *   - Pass `input_features` as an async generator that yields the first chunk's features then hop-based
 *     subsequent chunks; the model pulls from it and stops via its internal AudioExhaustedCriteria.
 *
 * The model emits each token ~NUM_DELAY_TOKENS behind the audio it is hearing, so for a fixed file the
 * final words never flush unless more audio follows them. We right-pad the clip with silence by the
 * processor's own offline amount (`num_right_pad_tokens` tokens) — the same flush its non-streaming
 * mode applies — before streaming, so the delayed tail tokens are emitted instead of truncated.
 * Streams partial text via onDelta; returns text + telemetry.
 */
export async function runVoxtralRealtime(
  handle: Extract<MultimodalHandle, { kind: "audiolm-stream" }>,
  audio: Float32Array,
  onDelta?: (text: string) => void,
): Promise<VlmResult> {
  const processor = handle.processor;
  const hopLength: number = processor.feature_extractor.config.hop_length;
  const winHalf = Math.floor(processor.feature_extractor.config.n_fft / 2);
  const firstChunkSamples: number = processor.num_samples_first_audio_chunk;
  const chunkSamples: number = processor.num_samples_per_audio_chunk;
  const firstMelFrames: number = processor.num_mel_frames_first_audio_chunk;
  const audioLenPerTok: number = processor.audio_length_per_tok;
  const rawPerTok: number = processor.raw_audio_length_per_tok;

  // Right-pad with silence to flush the transcription delay (see doc comment), matching the amount the
  // processor's own offline/non-streaming path uses. Without this the last word is truncated.
  const flushSamples = processor.num_right_pad_tokens * rawPerTok;
  const clip = new Float32Array(audio.length + flushSamples);
  clip.set(audio);

  // The first streaming chunk supplies the seed input_ids (BOS + NUM_LEFT_PAD + NUM_DELAY pad tokens).
  const firstChunk = await processor(clip.subarray(0, firstChunkSamples), { is_streaming: true, is_first_audio_chunk: true });

  // Async generator of per-chunk mel features that the streaming encoder consumes as generation runs.
  async function* featuresStream(): AsyncGenerator<unknown> {
    const first = await processor(clip.subarray(0, firstChunkSamples), { is_streaming: true, is_first_audio_chunk: true });
    yield first.input_features;
    let melFrameIdx = firstMelFrames;
    let startIdx = melFrameIdx * hopLength - winHalf;
    while (startIdx + chunkSamples < clip.length) {
      const endIdx = startIdx + chunkSamples;
      const chunk = await processor(clip.slice(startIdx, endIdx), { is_streaming: true, is_first_audio_chunk: false });
      yield chunk.input_features;
      melFrameIdx += audioLenPerTok;
      startIdx = melFrameIdx * hopLength - winHalf;
    }
  }

  // AudioExhaustedCriteria ends generation when the stream drains; this is only a safety backstop
  // sized to the clip (~one text token per raw_audio_length_per_tok samples).
  const maxNewTokens = Math.max(256, Math.ceil(clip.length / rawPerTok) + 128);

  const sink = { text: "", completionTokens: 0, firstTokenMs: undefined as number | undefined };
  const startedMs = performance.now();
  const streamer = makeCountingStreamer(handle.Streamer, processor.tokenizer, onDelta, sink);
  await handle.model.generate({
    input_ids: firstChunk.input_ids,
    input_features: featuresStream(),
    max_new_tokens: maxNewTokens,
    streamer,
  });
  return { text: sink.text.trim(), completionTokens: sink.completionTokens, firstTokenMs: sink.firstTokenMs, startedMs, endedMs: performance.now() };
}

/** Synthesize speech. Returns a RawAudio (Float32Array + sampling_rate + toBlob). */
export async function synthesize(handle: MultimodalHandle, text: string, opts: { voice: string; speed?: number }): Promise<RawAudio> {
  if (handle.kind === "tts-kokoro") return handle.tts.generate(text, { voice: opts.voice, speed: opts.speed ?? 1 });
  if (handle.kind === "tts-pipeline") {
    const synthOpts: AnyRecord = { speaker_embeddings: supertonicVoiceUrl(handle.modelId, opts.voice) };
    let input = text;
    if (handle.modelId === SUPERTONIC_2_MODEL_ID) {
      // Supertonic 2 is multilingual and REQUIRES the text wrapped in a language tag (e.g. <en>…</en>);
      // raw text produces partial words / repeats / gibberish. It also runs a diffusion denoiser, so
      // pass enough inference steps for clean audio. (Speed stays a playback-rate concern, as in v1.)
      input = `<en>${text}</en>`;
      synthOpts.num_inference_steps = 16;
    }
    return handle.synth(input, synthOpts);
  }
  throw new Error("Loaded model cannot synthesize speech.");
}

/** Whether a loaded TTS handle can stream sentence-by-sentence audio as text is pushed (Kokoro only). */
export function supportsStreamingTts(
  handle: MultimodalHandle | null | undefined,
): handle is Extract<MultimodalHandle, { kind: "tts-kokoro" }> {
  return handle?.kind === "tts-kokoro";
}

/**
 * Begin a streaming Kokoro synthesis. Push text into `splitter` as it arrives (e.g. from an LLM token
 * stream) and close() it when done; consume `chunks` to get one RawAudio per completed sentence. The
 * generator is pull-driven — synthesis is serialized per sentence and paced by the consumer loop — and
 * a splitter allows only ONE active iterator, so create a fresh pair per turn via this function.
 * @throws {Error} if the handle is not a Kokoro TTS handle.
 */
export function synthesizeStream(
  handle: MultimodalHandle,
  opts: { voice: string; speed?: number },
): { splitter: TtsTextSplitter; chunks: AsyncIterable<RawAudio> } {
  if (handle.kind !== "tts-kokoro") throw new Error("Streaming TTS requires a Kokoro model.");
  const splitter = new handle.TextSplitterStream();
  const stream = handle.tts.stream(splitter, { voice: opts.voice, speed: opts.speed ?? 1 });
  async function* chunks(): AsyncGenerator<RawAudio> {
    for await (const part of stream) yield part.audio;
  }
  return { splitter, chunks: chunks() };
}

/** Answer a question about an image (VLM). Streams partial text via onDelta; returns text + telemetry. */
export async function describeImage(handle: MultimodalHandle, imageSrc: string, prompt: string, onDelta?: (text: string) => void): Promise<VlmResult> {
  if (handle.kind === "gemma") return runGemma(handle, { imageSrc, prompt }, onDelta);
  if (handle.kind !== "vlm") throw new Error("Loaded model cannot analyze images.");

  const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: prompt }] }];
  const text = handle.processor.apply_chat_template(messages, { add_generation_prompt: true });
  const src = handle.maxPixels ? await capSrc(imageSrc, handle.maxPixels) : imageSrc;
  const image = await handle.loadImage(src);
  const inputs = handle.imageArray ? await handle.processor(text, [image], {}) : await handle.processor(text, image);

  const sink = { text: "", completionTokens: 0, firstTokenMs: undefined as number | undefined };
  const startedMs = performance.now();
  const streamer = makeCountingStreamer(handle.Streamer, handle.processor.tokenizer, onDelta, sink);
  await handle.model.generate({ ...inputs, max_new_tokens: handle.maxNewTokens, do_sample: false, streamer });
  return { text: sink.text.trim(), completionTokens: sink.completionTokens, firstTokenMs: sink.firstTokenMs, startedMs, endedMs: performance.now() };
}

/** Gemma-4 text + image + audio. image/audio placeholders go before the text; audio is a 16 kHz Float32Array. */
export async function runGemma(
  handle: Extract<MultimodalHandle, { kind: "gemma" }>,
  input: { imageSrc?: string; audio?: Float32Array; prompt: string },
  onDelta?: (text: string) => void,
): Promise<VlmResult> {
  const content: Array<AnyRecord> = [];
  if (input.imageSrc) content.push({ type: "image" });
  if (input.audio) content.push({ type: "audio" });
  content.push({ type: "text", text: input.prompt });

  const prompt = handle.processor.apply_chat_template([{ role: "user", content }], { enable_thinking: false, add_generation_prompt: true });
  const image = input.imageSrc ? await handle.loadImage(input.imageSrc) : null;
  const inputs = await handle.processor(prompt, image, input.audio ?? null, { add_special_tokens: false });

  const sink = { text: "", completionTokens: 0, firstTokenMs: undefined as number | undefined };
  const startedMs = performance.now();
  const streamer = makeCountingStreamer(handle.Streamer, handle.processor.tokenizer, onDelta, sink);
  await handle.model.generate({ ...inputs, max_new_tokens: 1024, do_sample: false, streamer });
  return { text: sink.text.trim(), completionTokens: sink.completionTokens, firstTokenMs: sink.firstTokenMs, startedMs, endedMs: performance.now() };
}

/** A chat turn as callers supply it (system/user/assistant + plain-text content). */
export type TextTurn = { role: string; content: string };

/**
 * Normalize chat messages into Gemma-friendly turns. Gemma's chat template has no dedicated
 * system role, so any leading system text is folded into the first user turn; each turn's content
 * becomes a `[{type:"text"}]` part (the same content shape the verified image/audio path uses).
 */
function toGemmaTurns(messages: TextTurn[]): Array<{ role: string; content: Array<AnyRecord> }> {
  const turns: Array<{ role: string; content: Array<AnyRecord> }> = [];
  let systemPrefix = "";
  for (const message of messages) {
    if (message.role === "system") {
      systemPrefix += (systemPrefix ? "\n\n" : "") + message.content;
      continue;
    }
    let text = message.content;
    if (message.role === "user" && systemPrefix) {
      text = `${systemPrefix}\n\n${text}`;
      systemPrefix = "";
    }
    turns.push({ role: message.role, content: [{ type: "text", text }] });
  }
  // System-only (no following user turn) still needs to be sent as a user message.
  if (systemPrefix) turns.unshift({ role: "user", content: [{ type: "text", text: systemPrefix }] });
  return turns;
}

/**
 * Gemma-4 text-only generation from a full chat-message list. Prompt-only (no constrained decoding),
 * matching the Transformers.js text path — a JSON schema is conveyed through the prompt, not a
 * grammar. Streams partial text via onDelta and returns text + real token/latency telemetry.
 */
export async function generateGemmaText(
  handle: Extract<MultimodalHandle, { kind: "gemma" }>,
  messages: TextTurn[],
  opts: { maxNewTokens?: number; temperature?: number; topP?: number; repetitionPenalty?: number; onDelta?: (text: string) => void } = {},
): Promise<VlmResult> {
  const prompt = handle.processor.apply_chat_template(toGemmaTurns(messages), { enable_thinking: false, add_generation_prompt: true });
  const inputs = await handle.processor(prompt, null, null, { add_special_tokens: false });

  const sink = { text: "", completionTokens: 0, firstTokenMs: undefined as number | undefined };
  const startedMs = performance.now();
  const streamer = makeCountingStreamer(handle.Streamer, handle.processor.tokenizer, opts.onDelta, sink);

  const doSample = (opts.temperature ?? 0) > 0;
  const genOptions: AnyRecord = { ...inputs, max_new_tokens: opts.maxNewTokens ?? 1024, do_sample: doSample, streamer };
  if (doSample && opts.temperature !== undefined) genOptions.temperature = opts.temperature;
  if (opts.topP !== undefined) genOptions.top_p = opts.topP;
  if (opts.repetitionPenalty !== undefined) genOptions.repetition_penalty = opts.repetitionPenalty;
  await handle.model.generate(genOptions);

  return { text: sink.text.trim(), completionTokens: sink.completionTokens, firstTokenMs: sink.firstTokenMs, startedMs, endedMs: performance.now() };
}

/* ------------------------------------------------------------------ */
/* Image helper                                                        */
/* ------------------------------------------------------------------ */

function loadHTMLImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image load failed"));
    img.src = src;
  });
}

/**
 * Downscale (via a plain canvas) so total pixels ≤ maxPixels, preserving aspect. Returns a PNG
 * data-URL. Guards the Qwen-family vision encoder against an int32 overflow on large images.
 * Browser main-thread only (`Image` + `document.createElement("canvas")` — not available in workers).
 * NOTE: RawImage.resize() is deliberately avoided — its -1 aspect arg produced a malformed image.
 */
export async function capSrc(src: string, maxPixels: number): Promise<string> {
  if (!maxPixels) return src;
  const img = await loadHTMLImage(src);
  const px = img.naturalWidth * img.naturalHeight;
  if (px <= maxPixels) return src;
  const scale = Math.sqrt(maxPixels / px);
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d")?.drawImage(img, 0, 0, w, h);
  return canvas.toDataURL("image/png");
}
