import type { ModelRecord } from "@mlc-ai/web-llm";

export type RuntimePreset = {
  maxTokens: number;
  temperature: number;
  topP: number;
  repetitionPenalty: number;
  frequencyPenalty: number;
  presencePenalty: number;
  seed: number | null;
  disableThinking: boolean;
};

export type ModelBackend = "webllm" | "transformers-js";

/** Input/output modality of a model, used for the library modality-flow line + Playground gating. */
export type Modality = "text" | "image" | "audio";
/** Capability task keys. Map to the TYPE chips: text→LLM, vision→VLM, stt→STT, tts→TTS. */
export type ModelTask = "text" | "vision" | "stt" | "tts";
/**
 * How a non-LLM-text model loads + runs in the browser. Undefined means a plain text LLM that uses
 * the existing WebLLM / Transformers.js text-generation path. Each variant maps to a verified
 * inference recipe in llm/multimodal.ts.
 */
export type MultimodalRuntime =
  | "stt" // automatic-speech-recognition pipeline (Whisper / Moonshine)
  | "tts-kokoro" // kokoro-js KokoroTTS
  | "tts-pipeline" // native text-to-speech pipeline (Supertonic)
  | "vlm-vision2seq" // AutoModelForVision2Seq (SmolVLM)
  | "vlm-imagetext" // AutoModelForImageTextToText (Qwen3-VL, GLM-OCR)
  | "audio-text-to-text" // AutoModelForAudioTextToText with an <audio> token in the prompt (Granite Speech, Ultravox, non-realtime Voxtral)
  | "voxtral-realtime" // VoxtralRealtimeForConditionalGeneration — audio-only streaming processor + generate({ input_features })
  | "gemma-mm"; // Gemma4ForConditionalGeneration (text + image + audio)

export type ModelPreset = {
  id: string;
  label: string;
  shortName: string;
  description: string;
  approximateDownload: string;
  vramRequiredMB: number;
  /** "estimated" means the value is not from WebLLM's bundled prebuilt metadata. */
  vramSource: "webllm" | "estimated";
  lowResourceRequired: boolean;
  disableThinkingSupported: boolean;
  defaultRuntime: RuntimePreset;
  source: "webllm-prebuilt" | "custom-mlc" | "transformers-onnx";
  backend: ModelBackend;
  stability: "recommended" | "stable" | "experimental";
  officialRepo: string;
  artifactRepo?: string;
  notes?: string;
  /** Optional dtype passed to Transformers.js, e.g. q4 or q4f16. */
  transformersDtype?: string;
  /** Capability tasks (TYPE chips + Playground tab gating). Defaults to ["text"] (a plain LLM). */
  tasks?: ModelTask[];
  /** Input modalities for the library modality-flow line. Defaults to ["text"]. */
  modalityIn?: Modality[];
  /** Output modalities. Defaults to ["text"]. */
  modalityOut?: Modality[];
  /** How a multimodal (non-LLM-text) model loads + runs. Undefined for plain text LLMs. */
  mmRuntime?: MultimodalRuntime;
  /** Optional display brand for the vendor tag (e.g. "Moonshine", "Kokoro", "Zhipu"). Falls back to the vendor label. */
  brand?: string;
};

/** Friendly capability names surfaced across the app (type chips/filters, setup dropdown/filters). */
export const TASK_LABELS: Record<ModelTask, string> = { text: "Text", vision: "Image", stt: "Transcription", tts: "Speech" };
export const TASK_ICONS: Record<ModelTask, string> = {
  text: "mdi-text-box-outline",
  vision: "mdi-image-outline",
  stt: "mdi-microphone-outline",
  tts: "mdi-account-voice",
};
export const MODALITY_LABELS: Record<Modality, string> = { text: "Text", image: "Image", audio: "Audio" };
export const MODALITY_ICONS: Record<Modality, string> = { text: "mdi-text", image: "mdi-image-outline", audio: "mdi-waveform" };

/** Task capabilities for a preset, defaulting a bare LLM to ["text"]. */
export function presetTasks(preset: ModelPreset): ModelTask[] {
  return preset.tasks && preset.tasks.length > 0 ? preset.tasks : ["text"];
}

/** Canonical ordering of the four capability slots (Text / Image / Transcription / Speech). */
export const SLOT_ORDER: ModelTask[] = ["text", "vision", "stt", "tts"];

/**
 * The capability SLOTS a model occupies when loaded — one per its {@link presetTasks}. Used to bound
 * concurrent loads to one model per slot (Text/Image/Transcription/Speech).
 *
 * NOTE: image-text-to-text VLMs (SmolVLM/Qwen3-VL/GLM-OCR) do NOT fill the Text slot — their
 * Transformers.js processors require an image and error ("No images provided") on text-only input, so
 * they cannot drive the text tabs. Only genuine text runtimes (LLMs, and Gemma-4 multimodal which has
 * a verified text path) fill Text. A VLM therefore occupies just the Image slot.
 */
export function modelSlots(preset: ModelPreset): ModelTask[] {
  const slots = new Set<ModelTask>(presetTasks(preset));
  return SLOT_ORDER.filter((task) => slots.has(task));
}
export function presetModalityIn(preset: ModelPreset): Modality[] {
  return preset.modalityIn && preset.modalityIn.length > 0 ? preset.modalityIn : ["text"];
}
export function presetModalityOut(preset: ModelPreset): Modality[] {
  return preset.modalityOut && preset.modalityOut.length > 0 ? preset.modalityOut : ["text"];
}

/** A selectable TTS speaker voice (id → display name + short tag). */
export type TtsVoice = { id: string; label: string; tag: string };

/** Model ids of the TTS presets, used to look up their voice rosters. */
export const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
export const SUPERTONIC_MODEL_ID = "onnx-community/Supertonic-TTS-ONNX";
export const SUPERTONIC_2_MODEL_ID = "onnx-community/Supertonic-TTS-2-ONNX";

/** Kokoro voice roster surfaced in the TTS tab (id → display name + tag), verified voice ids. */
export const KOKORO_VOICES: TtsVoice[] = [
  { id: "af_heart", label: "Heart", tag: "US female" },
  { id: "af_bella", label: "Bella", tag: "US female" },
  { id: "am_michael", label: "Michael", tag: "US male" },
  { id: "am_fenrir", label: "Fenrir", tag: "US male" },
  { id: "bf_emma", label: "Emma", tag: "UK female" },
  { id: "bm_george", label: "George", tag: "UK male" },
];

/**
 * Supertonic speaker roster. The repo ships ten speaker-embedding files under `voices/` — F1–F5
 * (female) and M1–M5 (male) — verified against the Hugging Face repo file tree. The voice id is the
 * `.bin` basename consumed as the pipeline's `speaker_embeddings`.
 */
export const SUPERTONIC_VOICES: TtsVoice[] = [
  { id: "F1", label: "Female 1", tag: "English" },
  { id: "F2", label: "Female 2", tag: "English" },
  { id: "F3", label: "Female 3", tag: "English" },
  { id: "F4", label: "Female 4", tag: "English" },
  { id: "F5", label: "Female 5", tag: "English" },
  { id: "M1", label: "Male 1", tag: "English" },
  { id: "M2", label: "Male 2", tag: "English" },
  { id: "M3", label: "Male 3", tag: "English" },
  { id: "M4", label: "Male 4", tag: "English" },
  { id: "M5", label: "Male 5", tag: "English" },
];

/** The selectable voices for a TTS model (empty for non-TTS models). */
export function ttsVoicesForModel(modelId: string): TtsVoice[] {
  if (modelId === KOKORO_MODEL_ID) return KOKORO_VOICES;
  if (modelId === SUPERTONIC_MODEL_ID || modelId === SUPERTONIC_2_MODEL_ID) return SUPERTONIC_VOICES;
  return [];
}

function extractionPreset(overrides: Partial<RuntimePreset> = {}): RuntimePreset {
  return {
    maxTokens: 512,
    temperature: 0,
    topP: 0.85,
    repetitionPenalty: 1.08,
    frequencyPenalty: 0.2,
    presencePenalty: 0,
    seed: 42,
    disableThinking: false,
    ...overrides,
  };
}

const gemmaFixedContextOverrides: NonNullable<ModelRecord["overrides"]> = {
  // Gemma 3 and current experimental Gemma 4 MLC configs can expose both values as positive
  // (for example context_window_size=4096 and sliding_window_size=512). WebLLM requires one
  // KV-cache strategy at load time, so use the fixed context-window path for this demo.
  context_window_size: 4096,
  sliding_window_size: -1,
  max_history_size: 1,
};

export const MODEL_RECORD_OVERRIDES: Record<string, NonNullable<ModelRecord["overrides"]>> = {
  "gemma3-1b-it-q4f16_1-MLC": gemmaFixedContextOverrides,
  "gemma-4-E2B-it-q4f16_1-MLC": gemmaFixedContextOverrides,
  "gemma-4-E4B-it-q4f16_1-MLC": gemmaFixedContextOverrides,
};

export function applyCuratedModelRecordOverrides(record: ModelRecord): ModelRecord {
  const overrides = MODEL_RECORD_OVERRIDES[record.model_id];
  if (!overrides) return record;

  return {
    ...record,
    overrides: {
      ...(record.overrides ?? {}),
      ...overrides,
    },
  };
}

// Gemma 4 is not bundled in @mlc-ai/web-llm 0.2.84's prebuiltAppConfig.
// WebLLM supports custom MLC model records, so these third-party records are appended
// to the runtime AppConfig and their artifacts are proxied through the Cloudflare Worker.
// Keep these marked experimental until they are tested on your target hardware.
export const CUSTOM_MODEL_RECORDS: ModelRecord[] = [
  {
    model: "https://huggingface.co/welcoma/gemma-4-E2B-it-q4f16_1-MLC",
    model_id: "gemma-4-E2B-it-q4f16_1-MLC",
    model_lib:
      "https://huggingface.co/welcoma/gemma-4-E2B-it-q4f16_1-MLC/resolve/main/libs/gemma-4-E2B-it-q4f16_1-MLC-webgpu.wasm",
    vram_required_MB: 4096,
    low_resource_required: false,
    required_features: ["shader-f16"],
    overrides: gemmaFixedContextOverrides,
  },
  {
    model: "https://huggingface.co/welcoma/gemma-4-E4B-it-q4f16_1-MLC",
    model_id: "gemma-4-E4B-it-q4f16_1-MLC",
    model_lib:
      "https://huggingface.co/welcoma/gemma-4-E4B-it-q4f16_1-MLC/resolve/main/libs/gemma-4-E4B-it-q4f16_1-MLC-webgpu.wasm",
    vram_required_MB: 6144,
    low_resource_required: false,
    required_features: ["shader-f16"],
    overrides: gemmaFixedContextOverrides,
  },
];

export const CUSTOM_MODEL_RECORD_IDS = new Set(CUSTOM_MODEL_RECORDS.map((record) => record.model_id));

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "Qwen3.5-0.8B-q4f16_1-MLC",
    label: "Qwen3.5 0.8B q4f16 — recommended tested default (~1.6 GB VRAM)",
    shortName: "Qwen3.5 0.8B",
    description:
      "Default browser-extraction model. It is the smallest recent WebLLM Qwen3.5 option and has worked well for deterministic JSON extraction in this app.",
    approximateDownload: "~447 MB model files plus browser cache overhead",
    vramRequiredMB: 1629.49,
    vramSource: "webllm",
    lowResourceRequired: true,
    disableThinkingSupported: true,
    defaultRuntime: extractionPreset({ disableThinking: true }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "recommended",
    officialRepo: "https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC",
  },
  {
    id: "Qwen3.5-2B-q4f16_1-MLC",
    label: "Qwen3.5 2B q4f16 — better quality, moderate memory (~2.2 GB VRAM)",
    shortName: "Qwen3.5 2B",
    description:
      "Larger Qwen3.5 option for better instruction following and JSON extraction when the device has enough GPU memory.",
    approximateDownload: "larger multi-file model download plus browser cache overhead",
    vramRequiredMB: 2245.44,
    vramSource: "webllm",
    lowResourceRequired: false,
    disableThinkingSupported: true,
    defaultRuntime: extractionPreset({ maxTokens: 640, disableThinking: true }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/Qwen3.5-2B-q4f16_1-MLC",
  },
  {
    id: "Qwen3.5-4B-q4f16_1-MLC",
    label: "Qwen3.5 4B q4f16 — higher quality desktop option (~3.8 GB VRAM)",
    shortName: "Qwen3.5 4B",
    description:
      "Higher-quality Qwen3.5 option for desktop GPUs. Use when extraction quality is more important than load time and memory footprint.",
    approximateDownload: "large multi-file model download plus browser cache overhead",
    vramRequiredMB: 3867.82,
    vramSource: "webllm",
    lowResourceRequired: false,
    disableThinkingSupported: true,
    defaultRuntime: extractionPreset({ maxTokens: 768, disableThinking: true, repetitionPenalty: 1.06 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/Qwen3.5-4B-q4f16_1-MLC",
  },
  {
    id: "gemma3-1b-it-q4f16_1-MLC",
    label: "Gemma 3 1B IT q4f16 — small Google fallback (~711 MB VRAM)",
    shortName: "Gemma 3 1B",
    description:
      "Small official WebLLM Gemma 3 model. This build uses a fixed-context override in the app so WebLLM does not try to enable both context-window and sliding-window KV-cache modes.",
    approximateDownload: "small model download plus browser cache overhead",
    vramRequiredMB: 711.07,
    vramSource: "webllm",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 512, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/gemma3-1b-it-q4f16_1-MLC",
  },
  {
    id: "gemma-4-E2B-it-q4f16_1-MLC",
    label: "Gemma 4 E2B IT q4f16 — experimental custom WebLLM build (~4 GB VRAM est.)",
    shortName: "Gemma 4 E2B",
    description:
      "Smallest Gemma 4 instruction option available here. This uses a third-party custom MLC/WebLLM artifact and a fixed-context override for WebLLM KV-cache loading.",
    approximateDownload: "~2.7 GB model repo plus browser cache overhead",
    vramRequiredMB: 4096,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "custom-mlc",
    backend: "webllm",
    stability: "experimental",
    officialRepo: "https://huggingface.co/google/gemma-4-E2B-it",
    artifactRepo: "https://huggingface.co/welcoma/gemma-4-E2B-it-q4f16_1-MLC",
  },
  {
    id: "gemma-4-E4B-it-q4f16_1-MLC",
    label: "Gemma 4 E4B IT q4f16 — experimental larger Gemma 4 build (~6 GB VRAM est.)",
    shortName: "Gemma 4 E4B",
    description:
      "Larger Gemma 4 instruction option for desktop/high-memory testing. This uses the same fixed-context WebLLM load override as the E2B option.",
    approximateDownload: "~4.3 GB model repo plus browser cache overhead",
    vramRequiredMB: 6144,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "custom-mlc",
    backend: "webllm",
    stability: "experimental",
    officialRepo: "https://huggingface.co/google/gemma-4-E4B-it",
    artifactRepo: "https://huggingface.co/welcoma/gemma-4-E4B-it-q4f16_1-MLC",
  },
  {
    id: "onnx-community/whisper-base",
    label: "Whisper Base — multilingual speech-to-text via Transformers.js/WebGPU",
    shortName: "Whisper Base",
    description: "OpenAI Whisper base — dependable multilingual speech-to-text with punctuation and segment timestamps.",
    approximateDownload: "~145 MB (encoder fp32 + q4/q8 decoder) plus browser cache overhead",
    vramRequiredMB: 500,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "recommended",
    officialRepo: "https://huggingface.co/onnx-community/whisper-base",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "stt",
    brand: "OpenAI",
  },
  {
    id: "onnx-community/whisper-tiny",
    label: "Whisper Tiny — smallest multilingual speech-to-text",
    shortName: "Whisper Tiny",
    description: "Smallest Whisper. Ultra-fast multilingual transcription for low-memory devices.",
    approximateDownload: "~41 MB q8 ONNX plus browser cache overhead",
    vramRequiredMB: 250,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "stable",
    officialRepo: "https://huggingface.co/onnx-community/whisper-tiny",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "stt",
    brand: "OpenAI",
  },
  {
    id: "onnx-community/moonshine-base-ONNX",
    label: "Moonshine Base — fast English on-device speech-to-text",
    shortName: "Moonshine Base",
    description: "Fast English speech-to-text tuned for short, real-time clips and low-latency on-device transcription.",
    approximateDownload: "~63 MB q8 ONNX plus browser cache overhead",
    vramRequiredMB: 400,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "stable",
    officialRepo: "https://huggingface.co/onnx-community/moonshine-base-ONNX",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "stt",
    brand: "Moonshine",
  },
  {
    id: "onnx-community/moonshine-tiny-ONNX",
    label: "Moonshine Tiny — smallest fast English on-device speech-to-text",
    shortName: "Moonshine Tiny",
    description: "Smallest Moonshine. Ultra-low-latency English speech-to-text for short, real-time clips on low-memory devices.",
    approximateDownload: "~75 MB (fp32 encoder + q4 decoder) plus browser cache overhead",
    vramRequiredMB: 250,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "stable",
    officialRepo: "https://huggingface.co/onnx-community/moonshine-tiny-ONNX",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "stt",
    brand: "Moonshine",
  },
  {
    id: "onnx-community/whisper-medium_timestamped",
    label: "Whisper Medium — higher-accuracy multilingual speech-to-text (word timestamps)",
    shortName: "Whisper Medium",
    description:
      "OpenAI Whisper medium (~769M) with word-level timestamps. Stronger multilingual accuracy than Base/Tiny, at a much larger download and slower first load.",
    approximateDownload: "~1.7 GB (fp32 encoder + q4 decoder) plus browser cache overhead",
    vramRequiredMB: 1800,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/onnx-community/whisper-medium_timestamped",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "stt",
    brand: "OpenAI",
  },
  {
    id: "onnx-community/whisper-medium.en_timestamped",
    label: "Whisper Medium English — English-only medium speech-to-text (word timestamps)",
    shortName: "Whisper Medium EN",
    description:
      "English-only Whisper medium (~769M) with word-level timestamps. Tuned for English accuracy; do not use it for other languages.",
    approximateDownload: "~1.7 GB (fp32 encoder + q4 decoder) plus browser cache overhead",
    vramRequiredMB: 1800,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/onnx-community/whisper-medium.en_timestamped",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "stt",
    brand: "OpenAI",
  },
  {
    id: "onnx-community/cohere-transcribe-03-2026-ONNX",
    label: "Cohere Transcribe — multilingual speech-to-text via Transformers.js/WebGPU",
    shortName: "Cohere Transcribe",
    description:
      "Cohere's multilingual transcription model (14 languages). Loads via the automatic-speech-recognition pipeline with a flat q4 dtype — its fp32 encoder is multiple GB, so q4 is the documented WebGPU recipe.",
    approximateDownload: "~2.1 GB q4 ONNX (q4 encoder + q4 decoder) plus browser cache overhead",
    vramRequiredMB: 2600,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX",
    transformersDtype: "q4",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "stt",
    brand: "Cohere",
  },
  {
    id: "onnx-community/parakeet-ctc-0.6b-ONNX",
    label: "Parakeet CTC 0.6B — fast low-latency English speech-to-text (NVIDIA FastConformer)",
    shortName: "Parakeet CTC 0.6B",
    description:
      "NVIDIA Parakeet CTC 0.6B (FastConformer). Fast, low-latency English transcription through the CTC speech-recognition pipeline. Loads with a flat q4f16 dtype (single-module CTC model — no separate encoder/decoder).",
    approximateDownload: "~450 MB q4f16 ONNX plus browser cache overhead",
    vramRequiredMB: 1200,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX",
    transformersDtype: "q4f16",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "stt",
  },
  {
    id: "onnx-community/granite-speech-4.1-2b-ONNX",
    label: "IBM Granite Speech 4.1 2B — audio-LLM speech-to-text via Transformers.js/WebGPU",
    shortName: "Granite Speech 2B",
    description:
      "IBM Granite Speech (2B): a speech-aware LLM (Conformer encoder + Granite LLM + audio LoRA) with strong English ASR. Runs through the audio-text-to-text recipe — it generates the transcript, so there are no per-segment timestamps.",
    approximateDownload: "~1.5 GB q4f16 ONNX (audio encoder + decoder + embeddings) plus browser cache overhead",
    vramRequiredMB: 2200,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/onnx-community/granite-speech-4.1-2b-ONNX",
    transformersDtype: "q4f16",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "audio-text-to-text",
    brand: "IBM Granite",
  },
  {
    id: "onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX",
    label: "Voxtral Mini 4B Realtime — Mistral audio-LLM speech-to-text via Transformers.js/WebGPU",
    shortName: "Voxtral Mini 4B RT",
    description:
      "Mistral Voxtral Mini 4B: an audio-LLM for multilingual transcription (14 languages) built on Ministral. Runs through the audio-text-to-text recipe as a single-pass transcription. This is the realtime/streaming checkpoint and is heavy (4B), so single-pass output quality should be validated on your hardware.",
    approximateDownload: "~2.8 GB q4f16 ONNX (audio encoder + decoder + embeddings) plus browser cache overhead",
    vramRequiredMB: 4096,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602",
    artifactRepo: "https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX",
    transformersDtype: "q4f16",
    tasks: ["stt"],
    modalityIn: ["audio"],
    modalityOut: ["text"],
    mmRuntime: "voxtral-realtime",
  },
  {
    id: "onnx-community/Kokoro-82M-v1.0-ONNX",
    label: "Kokoro 82M — natural on-device text-to-speech (kokoro-js)",
    shortName: "Kokoro 82M",
    description: "Compact, natural text-to-speech with a library of expressive voices. A browser-TTS favorite.",
    approximateDownload: "~86 MB q8 (326 MB fp32 on WebGPU) plus browser cache overhead",
    vramRequiredMB: 420,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "recommended",
    officialRepo: "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX",
    tasks: ["tts"],
    modalityIn: ["text"],
    modalityOut: ["audio"],
    mmRuntime: "tts-kokoro",
    brand: "Kokoro",
    notes: "Loaded via kokoro-js from CDN; it bundles its own Transformers.js runtime and fetches weights directly from Hugging Face.",
  },
  {
    id: "onnx-community/Supertonic-TTS-ONNX",
    label: "Supertonic — low-latency streaming text-to-speech (native pipeline)",
    shortName: "Supertonic",
    description: "Very low-latency on-device text-to-speech built for real-time streaming playback.",
    approximateDownload: "~fp32 ONNX plus browser cache overhead",
    vramRequiredMB: 350,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/onnx-community/Supertonic-TTS-ONNX",
    tasks: ["tts"],
    modalityIn: ["text"],
    modalityOut: ["audio"],
    mmRuntime: "tts-pipeline",
    brand: "Supertonic",
  },
  {
    id: "onnx-community/Supertonic-TTS-2-ONNX",
    label: "Supertonic 2 — multilingual low-latency streaming text-to-speech (native pipeline)",
    shortName: "Supertonic 2",
    description:
      "Second-generation Supertonic: very low-latency on-device text-to-speech for real-time streaming playback, now multilingual (English, Korean, Spanish, Portuguese, French). Same ten F1–F5 / M1–M5 speaker voices as v1.",
    approximateDownload: "~260 MB fp32 ONNX plus browser cache overhead",
    vramRequiredMB: 400,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/onnx-community/Supertonic-TTS-2-ONNX",
    tasks: ["tts"],
    modalityIn: ["text"],
    modalityOut: ["audio"],
    mmRuntime: "tts-pipeline",
    brand: "Supertonic",
  },
  {
    id: "HuggingFaceTB/SmolVLM-256M-Instruct",
    label: "SmolVLM 256M — tiny vision-language model via Transformers.js/WebGPU",
    shortName: "SmolVLM 256M",
    description: "Tiny vision-language model for captioning and quick image Q&A. Runs on almost any device.",
    approximateDownload: "~190 MB q4f16 ONNX plus browser cache overhead",
    vramRequiredMB: 600,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "stable",
    officialRepo: "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct",
    transformersDtype: "q4",
    tasks: ["vision"],
    modalityIn: ["text", "image"],
    modalityOut: ["text"],
    mmRuntime: "vlm-vision2seq",
  },
  {
    id: "HuggingFaceTB/SmolVLM-500M-Instruct",
    label: "SmolVLM 500M — stronger small vision-language model",
    shortName: "SmolVLM 500M",
    description: "Small vision-language model with stronger scene and document understanding than the 256M build.",
    approximateDownload: "~250 MB q4f16 ONNX plus browser cache overhead",
    vramRequiredMB: 900,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "stable",
    officialRepo: "https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct",
    transformersDtype: "q4",
    tasks: ["vision"],
    modalityIn: ["text", "image"],
    modalityOut: ["text"],
    mmRuntime: "vlm-vision2seq",
  },
  {
    id: "onnx-community/Qwen3-VL-2B-Instruct-ONNX",
    label: "Qwen3-VL 2B — capable vision-language model via Transformers.js/WebGPU",
    shortName: "Qwen3-VL 2B",
    description: "Capable vision-language model for detailed image reasoning, charts, screenshots, and multi-step questions.",
    approximateDownload: "~1.5 GB mixed-precision ONNX plus browser cache overhead",
    vramRequiredMB: 2500,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
    artifactRepo: "https://huggingface.co/onnx-community/Qwen3-VL-2B-Instruct-ONNX",
    tasks: ["vision"],
    modalityIn: ["text", "image"],
    modalityOut: ["text"],
    mmRuntime: "vlm-imagetext",
  },
  {
    id: "wolfofbackstreet/GLM-OCR-ONNX-q4f16",
    label: "GLM-OCR — document OCR vision-language model via Transformers.js/WebGPU",
    shortName: "GLM-OCR",
    description: "Document-focused vision model for OCR, layout parsing, and structured text extraction from images.",
    approximateDownload: "~658 MB q4f16 ONNX plus browser cache overhead",
    vramRequiredMB: 1800,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/zai-org/GLM-OCR",
    artifactRepo: "https://huggingface.co/wolfofbackstreet/GLM-OCR-ONNX-q4f16",
    transformersDtype: "q4f16",
    tasks: ["vision"],
    modalityIn: ["text", "image"],
    modalityOut: ["text"],
    mmRuntime: "vlm-imagetext",
    brand: "Zhipu",
  },
  {
    id: "onnx-community/Qwen2.5-VL-3B-Instruct-ONNX",
    label: "Qwen2.5-VL 3B — vision-language model via Transformers.js/WebGPU",
    shortName: "Qwen2.5-VL 3B",
    description:
      "Qwen2.5-VL 3B for detailed image reasoning, charts, screenshots, and documents. Larger and stronger than the Qwen3-VL 2B build, at a heavier first download and higher GPU-memory needs.",
    approximateDownload: "~3.7 GB mixed-precision ONNX (fp16 vision + q4f16 decoder) plus browser cache overhead",
    vramRequiredMB: 3500,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
    artifactRepo: "https://huggingface.co/onnx-community/Qwen2.5-VL-3B-Instruct-ONNX",
    tasks: ["vision"],
    modalityIn: ["text", "image"],
    modalityOut: ["text"],
    mmRuntime: "vlm-imagetext",
  },
  {
    id: "onnx-community/LightOnOCR-2-1B-ONNX",
    label: "LightOnOCR-2 1B — document OCR vision-language model via Transformers.js/WebGPU",
    shortName: "LightOnOCR-2 1B",
    description:
      "Compact document-OCR vision model for text extraction, tables, forms, and layout parsing across 11 languages. Loads flat q4f16 like the GLM-OCR build.",
    approximateDownload: "~680 MB q4f16 ONNX plus browser cache overhead",
    vramRequiredMB: 1500,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset(),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/lightonai/LightOnOCR-2-1B",
    artifactRepo: "https://huggingface.co/onnx-community/LightOnOCR-2-1B-ONNX",
    transformersDtype: "q4f16",
    tasks: ["vision"],
    modalityIn: ["text", "image"],
    modalityOut: ["text"],
    mmRuntime: "vlm-imagetext",
    brand: "LightOn",
  },
  {
    id: "onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX",
    label: "NVIDIA Nemotron 3 Nano 4B ONNX — experimental Transformers.js/WebGPU (~2.5 GB download)",
    shortName: "Nemotron 3 Nano 4B",
    description:
      "Experimental ONNX + Transformers.js backend from the WebGPU demo space. It is prompt-only JSON generation, not WebLLM schema-constrained decoding, and may require substantially more disk/GPU memory than Qwen3.5.",
    approximateDownload: "~2.5 GB browser-selected quantized ONNX artifacts, cached after first load",
    vramRequiredMB: 4096,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: true,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.95, repetitionPenalty: 1.05, disableThinking: true }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    artifactRepo: "https://huggingface.co/onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX",
    transformersDtype: "q4",
    notes: "Loaded through @huggingface/transformers from CDN at runtime so the WebLLM demo remains small until this model is selected.",
  },
  {
    id: "LiquidAI/LFM2.5-230M-ONNX",
    label: "Liquid LFM2.5 230M ONNX — tiny Transformers.js/WebGPU model (~300 MB class)",
    shortName: "LFM2.5 230M",
    description:
      "Very small LiquidAI ONNX model for browser testing. It is useful when you want a fast, low-footprint Transformers.js/WebGPU baseline for extraction prompts.",
    approximateDownload: "~300 MB class model artifacts plus browser cache overhead",
    vramRequiredMB: 768,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 384, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/LiquidAI/LFM2.5-230M",
    artifactRepo: "https://huggingface.co/LiquidAI/LFM2.5-230M-ONNX",
    transformersDtype: "q4",
  },
  {
    id: "LiquidAI/LFM2.5-350M-ONNX",
    label: "Liquid LFM2.5 350M ONNX — compact Transformers.js/WebGPU model (~700 MB class)",
    shortName: "LFM2.5 350M",
    description:
      "Compact LiquidAI edge model with an official ONNX/WebGPU export. Good for comparing a hybrid architecture against Qwen, Gemma, and Llama in browser extraction.",
    approximateDownload: "~700 MB class model artifacts plus browser cache overhead",
    vramRequiredMB: 1024,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 448, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/LiquidAI/LFM2.5-350M",
    artifactRepo: "https://huggingface.co/LiquidAI/LFM2.5-350M-ONNX",
    transformersDtype: "q4",
  },
  {
    id: "LiquidAI/LFM2.5-1.2B-Thinking-ONNX",
    label: "Liquid LFM2.5 1.2B Thinking ONNX — reasoning-focused browser model",
    shortName: "LFM2.5 1.2B Thinking",
    description:
      "Reasoning-focused LiquidAI ONNX model for Transformers.js/WebGPU. It may emit <think> traces, so extraction prompts should explicitly request final JSON only.",
    approximateDownload: "large ONNX model artifacts plus browser cache overhead",
    vramRequiredMB: 2300,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: true,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.06, disableThinking: true }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking",
    artifactRepo: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX",
    transformersDtype: "q4",
  },
  {
    id: "onnx-community/Falcon-H1-Tiny-90M-Instruct-ONNX",
    label: "Falcon H1 Tiny 90M ONNX — ultra-small Transformers.js/WebGPU smoke-test model",
    shortName: "Falcon H1 Tiny 90M",
    description:
      "Ultra-small hybrid Mamba/attention model. It is best as a browser compatibility smoke test; extraction quality will likely be lower than the larger models.",
    approximateDownload: "~1.1 GB model repo, but selected quantized artifacts may be smaller",
    vramRequiredMB: 512,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 256, topP: 0.9, repetitionPenalty: 1.04 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct",
    artifactRepo: "https://huggingface.co/onnx-community/Falcon-H1-Tiny-90M-Instruct-ONNX",
    transformersDtype: "q4",
  },
  {
    id: "onnx-community/granite-4.0-micro-ONNX-web",
    label: "IBM Granite 4.0 Micro ONNX-web — small hybrid chat model via Transformers.js/WebGPU",
    shortName: "Granite 4.0 Micro ONNX",
    description:
      "Small IBM Granite MoE/hybrid chat model packaged specifically for Transformers.js/WebGPU. Useful as a non-Qwen/non-Llama browser extraction comparison.",
    approximateDownload: "ONNX-web model artifacts plus tokenizer/config files",
    vramRequiredMB: 2048,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 512, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/ibm-granite/granite-4.0-micro",
    artifactRepo: "https://huggingface.co/onnx-community/granite-4.0-micro-ONNX-web",
    transformersDtype: "q4f16",
  },
  {
    id: "onnx-community/granite-4.0-350m-ONNX-web",
    label: "IBM Granite 4.0 350M ONNX-web — small Transformers.js/WebGPU model",
    shortName: "Granite 4.0 350M",
    description:
      "Small Granite 4.0 ONNX-web build packaged for Transformers.js/WebGPU. This is one of the safest Granite test options for lower-memory machines.",
    approximateDownload: "~350 MB q4f16 ONNX artifacts plus tokenizer/config files",
    vramRequiredMB: 1024,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 448, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/ibm-granite/granite-4.0-350m",
    artifactRepo: "https://huggingface.co/onnx-community/granite-4.0-350m-ONNX-web",
    transformersDtype: "q4f16",
  },
  {
    id: "onnx-community/granite-4.0-h-350m-ONNX",
    label: "IBM Granite 4.0 H 350M ONNX — hybrid/MoE Transformers.js/WebGPU test",
    shortName: "Granite 4.0 H 350M",
    description:
      "Small Granite hybrid/MoE ONNX export. Use it to compare the Granite H family against Qwen3.5 and LiquidAI in browser extraction.",
    approximateDownload: "~236 MB q4f16 ONNX data plus graph/tokenizer/config files",
    vramRequiredMB: 1024,
    vramSource: "estimated",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 448, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/ibm-granite/granite-4.0-h-350m",
    artifactRepo: "https://huggingface.co/onnx-community/granite-4.0-h-350m-ONNX",
    transformersDtype: "q4f16",
  },
  {
    id: "onnx-community/granite-4.0-1b-ONNX-web",
    label: "IBM Granite 4.0 1B ONNX-web — larger Transformers.js/WebGPU model",
    shortName: "Granite 4.0 1B",
    description:
      "Granite 4.0 1B ONNX-web build packaged for Transformers.js/WebGPU. Expect better extraction quality than the 350M variant and a larger first download.",
    approximateDownload: "~1.25 GB q4f16 ONNX artifacts plus tokenizer/config files",
    vramRequiredMB: 2048,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 640, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/ibm-granite/granite-4.0-1b",
    artifactRepo: "https://huggingface.co/onnx-community/granite-4.0-1b-ONNX-web",
    transformersDtype: "q4f16",
  },
  {
    id: "onnx-community/granite-4.0-h-1b-ONNX",
    label: "IBM Granite 4.0 H 1B ONNX — hybrid/MoE Transformers.js/WebGPU model",
    shortName: "Granite 4.0 H 1B",
    description:
      "Larger Granite H hybrid/MoE ONNX export. Use this on machines that can handle around a 1 GB quantized ONNX download plus runtime memory.",
    approximateDownload: "~925 MB q4f16 ONNX data plus graph/tokenizer/config files",
    vramRequiredMB: 2048,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 640, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/ibm-granite/granite-4.0-h-1b",
    artifactRepo: "https://huggingface.co/onnx-community/granite-4.0-h-1b-ONNX",
    transformersDtype: "q4f16",
  },
  {
    id: "onnx-community/granite-4.0-h-micro-ONNX",
    label: "IBM Granite 4.0 H Micro ONNX — larger hybrid/MoE test",
    shortName: "Granite 4.0 H Micro",
    description:
      "Large Granite H Micro ONNX export. It is included for testing but is not a first-load choice because the repository contains very large model variants.",
    approximateDownload: "~1.95 GB q4f16 ONNX data plus graph/tokenizer/config files",
    vramRequiredMB: 4096,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/ibm-granite/granite-4.0-h-micro",
    artifactRepo: "https://huggingface.co/onnx-community/granite-4.0-h-micro-ONNX",
    transformersDtype: "q4f16",
  },
  {
    id: "Mike0021/MiniCPM5-1B-ONNX-Web",
    label: "MiniCPM5 1B ONNX-web — community q4 Transformers.js/WebGPU export",
    shortName: "MiniCPM5 1B ONNX-web",
    description:
      "Browser-friendly q4 Transformers.js export of MiniCPM5 1B. The official ONNX Runtime GenAI repo is linked as the official repo; this selected artifact repo is the browser-loadable q4 export.",
    approximateDownload: "~902 MB q4 ONNX artifact plus tokenizer/config files",
    vramRequiredMB: 2048,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: true,
    defaultRuntime: extractionPreset({ maxTokens: 640, topP: 0.9, repetitionPenalty: 1.05, disableThinking: true }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/onnx-community/MiniCPM5-1B",
    artifactRepo: "https://huggingface.co/Mike0021/MiniCPM5-1B-ONNX-Web",
    transformersDtype: "q4",
  },
  {
    id: "HuggingFaceTB/SmolLM3-3B-ONNX",
    label: "SmolLM3 3B ONNX — current compact 3B model via Transformers.js/WebGPU",
    shortName: "SmolLM3 3B ONNX",
    description:
      "Current SmolLM3 ONNX export for Transformers.js. It is larger than the default model but useful for testing a fully open 3B browser/local model path.",
    approximateDownload: "large ONNX model artifacts plus tokenizer/config files",
    vramRequiredMB: 3072,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/HuggingFaceTB/SmolLM3-3B",
    artifactRepo: "https://huggingface.co/HuggingFaceTB/SmolLM3-3B-ONNX",
    transformersDtype: "q4",
  },
  {
    id: "onnx-community/gemma-4-E2B-it-ONNX",
    label: "Gemma 4 E2B ONNX — experimental Transformers.js/WebGPU alternative",
    shortName: "Gemma 4 E2B ONNX",
    description:
      "ONNX/Transformers.js route for Gemma 4 E2B. This is separate from the custom WebLLM/MLC Gemma 4 build and may be useful if the MLC artifact is not stable on a device.",
    approximateDownload: "large ONNX model artifacts plus tokenizer/config files",
    vramRequiredMB: 4096,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/google/gemma-4-E2B-it",
    artifactRepo: "https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX",
    transformersDtype: "q4f16",
    tasks: ["text", "vision", "stt"],
    modalityIn: ["text", "image", "audio"],
    modalityOut: ["text"],
    mmRuntime: "gemma-mm",
  },
  {
    id: "onnx-community/gemma-4-E4B-it-ONNX",
    label: "Gemma 4 E4B ONNX — larger multimodal Transformers.js/WebGPU build",
    shortName: "Gemma 4 E4B ONNX",
    description:
      "Larger ONNX/Transformers.js route for Gemma 4 E4B (~8B). Text + image + audio in one model, with stronger quality than the E2B ONNX build at a heavier first load and higher GPU-memory needs.",
    approximateDownload: "~5–6 GB q4f16 ONNX artifacts plus tokenizer/config files",
    vramRequiredMB: 6144,
    vramSource: "estimated",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "transformers-onnx",
    backend: "transformers-js",
    stability: "experimental",
    officialRepo: "https://huggingface.co/google/gemma-4-E4B-it",
    artifactRepo: "https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX",
    transformersDtype: "q4f16",
    tasks: ["text", "vision", "stt"],
    modalityIn: ["text", "image", "audio"],
    modalityOut: ["text"],
    mmRuntime: "gemma-mm",
  },
  {
    id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    label: "Llama 3.2 1B Instruct q4f16 — low-memory Meta fallback (~879 MB VRAM)",
    shortName: "Llama 3.2 1B",
    description:
      "Compact Meta Llama instruction model. It is a useful compatibility fallback and gives you another model family to compare against Qwen3.5.",
    approximateDownload: "small model download plus browser cache overhead",
    vramRequiredMB: 879.04,
    vramSource: "webllm",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 448, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC",
  },
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    label: "Llama 3.2 3B Instruct q4f16 — stronger Meta fallback (~2.2 GB VRAM)",
    shortName: "Llama 3.2 3B",
    description: "More capable Llama option for machines that can handle a larger local browser model.",
    approximateDownload: "large model download plus browser cache overhead",
    vramRequiredMB: 2263.69,
    vramSource: "webllm",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
  },
  {
    id: "Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
    label: "Hermes 3 Llama 3.2 3B q4f16 — extraction/chat alternate (~2.2 GB VRAM)",
    shortName: "Hermes 3 Llama 3.2 3B",
    description:
      "Hermes-tuned Llama 3.2 3B variant. Useful for comparing structured-output behavior against the base Llama instruct build.",
    approximateDownload: "large model download plus browser cache overhead",
    vramRequiredMB: 2263.69,
    vramSource: "webllm",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.06 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
  },
  {
    id: "OLMo-2-0425-1B-Instruct-q4f16_1-MLC",
    label: "OLMo 2 0425 1B Instruct q4f16 — open 1B alternate (~1.7 GB VRAM)",
    shortName: "OLMo 2 1B",
    description:
      "Recent small OLMo 2 instruction model compiled for WebLLM. Good for a non-Qwen, non-Llama comparison while staying near the 1B class.",
    approximateDownload: "medium model download plus browser cache overhead",
    vramRequiredMB: 1776.75,
    vramSource: "webllm",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 512, topP: 0.9, repetitionPenalty: 1.06 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/OLMo-2-0425-1B-Instruct-q4f16_1-MLC",
  },
  {
    id: "Phi-4-mini-instruct-q4f16_1-MLC",
    label: "Phi-4 mini Instruct q4f16 — capable desktop option (~3.4 GB VRAM)",
    shortName: "Phi-4 mini",
    description:
      "Capable compact Phi model. Use on desktop-class devices when smaller models miss fields or produce lower-quality extraction.",
    approximateDownload: "large model download plus browser cache overhead",
    vramRequiredMB: 3437.58,
    vramSource: "webllm",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/Phi-4-mini-instruct-q4f16_1-MLC",
  },
  {
    id: "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC",
    label: "Ministral 3 3B Instruct q4f16 — recent Mistral-family option (~2.8 GB VRAM)",
    shortName: "Ministral 3 3B",
    description:
      "Recent Mistral-family WebLLM option. Useful for comparing extraction quality against Qwen3.5/Llama/Phi on desktop-class GPUs.",
    approximateDownload: "large model download plus browser cache overhead",
    vramRequiredMB: 2863.69,
    vramSource: "webllm",
    lowResourceRequired: true,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC",
  },
  {
    id: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    label: "Mistral 7B Instruct v0.3 q4f16 — high-memory quality test (~4.5 GB VRAM)",
    shortName: "Mistral 7B v0.3",
    description:
      "Larger Mistral-family model. Keep this as a desktop/high-memory comparison model; it is not a first-load choice for casual users.",
    approximateDownload: "very large model download plus browser cache overhead",
    vramRequiredMB: 4573.39,
    vramSource: "webllm",
    lowResourceRequired: false,
    disableThinkingSupported: false,
    defaultRuntime: extractionPreset({ maxTokens: 768, topP: 0.9, repetitionPenalty: 1.05 }),
    source: "webllm-prebuilt",
    backend: "webllm",
    stability: "stable",
    officialRepo: "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
  },
];

export function getModelPreset(modelId: string): ModelPreset {
  return MODEL_PRESETS.find((preset) => preset.id === modelId) ?? MODEL_PRESETS[0];
}

export function formatMB(mb: number): string {
  if (!Number.isFinite(mb)) return "unknown";
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${Math.round(mb)} MB`;
}

export function modelRepoLabel(preset: ModelPreset): string {
  try {
    const url = new URL(preset.artifactRepo ?? preset.officialRepo);
    return url.pathname.replace(/^\//, "");
  } catch {
    return preset.id;
  }
}
