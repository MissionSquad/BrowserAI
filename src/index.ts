/**
 * @missionsquad/browserai — run small open AI models entirely in the browser with WebGPU.
 *
 * Core entry points:
 * - {@link BrowserAI} — model catalog + capability-slot manager + inference façade.
 * - {@link VoicePipeline} — headless press-to-talk voice assistant (mic → STT → LLM → TTS).
 */

// Client
export { BrowserAI, buildModelCacheTarget, normalizeContextLength } from "./client.js";
export type { BrowserAIEvents, GenerateTextOptions, LoadOptions, LoadProgress, LoadedModel, StreamingTtsSession, TtsOverrides } from "./client.js";

// Configuration
export { DEFAULT_CONTEXT_LENGTH, resolveConfig } from "./config.js";
export type { BrowserAIConfig, HistoryOptions, KokoroOptions, ResolvedConfig, StorageLike, TtsDefaults, WebLLMOptions } from "./config.js";

// Errors
export {
  BrowserAIError,
  formatError,
  MicrophoneError,
  MissingModelError,
  ModelLoadError,
  ProxyVerificationError,
  UnknownModelError,
  VoiceTurnError,
  WebGPUUnavailableError,
} from "./errors.js";

// Events
export { TypedEmitter } from "./events.js";
export type { EventMap, Listener } from "./events.js";

// Model catalog
export {
  applyCuratedModelRecordOverrides,
  CUSTOM_MODEL_RECORD_IDS,
  CUSTOM_MODEL_RECORDS,
  formatMB,
  getModelPreset,
  KOKORO_MODEL_ID,
  KOKORO_VOICES,
  MODALITY_ICONS,
  MODALITY_LABELS,
  MODEL_PRESETS,
  MODEL_RECORD_OVERRIDES,
  modelRepoLabel,
  modelSlots,
  presetModalityIn,
  presetModalityOut,
  presetTasks,
  SLOT_ORDER,
  SUPERTONIC_2_MODEL_ID,
  SUPERTONIC_MODEL_ID,
  SUPERTONIC_VOICES,
  TASK_ICONS,
  TASK_LABELS,
  ttsVoicesForModel,
} from "./models.js";
export type { Modality, ModelBackend, ModelPreset, ModelTask, MultimodalRuntime, RuntimePreset, TtsVoice } from "./models.js";

// Generation parameters + JSON mode
export { buildResponseFormat, clampNumber, parseOptionalInteger, runtimeParamsFromPreset } from "./generation.js";
export type { JsonMode, RuntimeParameters } from "./generation.js";

// Text inference
export { buildRuntimeStatsText, collectGenericStats, EMPTY_RESPONSE, generate, normalizeLatencyBreakdown } from "./inference.js";
export type {
  ChatMessage,
  EngineHandles,
  GenerateOptions,
  GenerationResult,
  LatencyBreakdown,
  RuntimeStats,
  TransformersModule,
  TransformersTextGenerationPipeline,
  TransformersTokenizer,
  WebLLMUsageExtra,
} from "./inference.js";

// Multimodal runtimes (STT / TTS / VLM / Gemma)
export {
  capSrc,
  DEFAULT_KOKORO_CDN,
  describeImage,
  disposeMultimodal,
  generateGemmaText,
  loadMultimodal,
  planGemmaAudioWindows,
  runAudioLm,
  runGemma,
  runVoxtralRealtime,
  supertonicVoiceUrl,
  supportsStreamingTts,
  synthesize,
  synthesizeStream,
  transcribe,
} from "./multimodal.js";
export type {
  AudioWindow,
  GenStats,
  KokoroModule,
  LoadMultimodalOptions,
  MultimodalDevice,
  MultimodalHandle,
  MultimodalTransformers,
  ProgressCallback,
  RawAudio,
  SttResult,
  TextTurn,
  TtsTextSplitter,
  VlmResult,
} from "./multimodal.js";

// JSON helpers + schema validation
export { detectLikelyRepetition, extractJsonCandidate, normalizeModelText, parseJsonFromModel, prettyJson } from "./json.js";
export type { JsonParseResult } from "./json.js";
export { validateAgainstJsonSchema } from "./schema-validator.js";
export { DEFAULT_EXTRACTION_SCHEMA, stringifySchema, stripSchemaMetadata } from "./schema.js";
export { buildExtractionMessages, DEFAULT_EXTRACTION_INSTRUCTIONS } from "./prompt.js";

// Model source (direct vs proxy) + proxy verification
export {
  applyContextOverride,
  buildProxiedAppConfig,
  buildWebLLMAppConfig,
  modelSourceDescription,
  proxyHuggingFaceUrl,
  proxyModelLibraryUrl,
  proxyModelRecord,
  proxyRawGithubUrl,
  TRANSFORMERS_CACHE_KEY,
  withCustomModelRecords,
} from "./model-source.js";
export type { BrowserCacheBackend, ModelSourceMode } from "./model-source.js";
export { splitHuggingFaceRepoId, verifyTransformersProxy, verifyWebLLMProxy } from "./proxy-verify.js";

// Hardware probing
export { collectHardwareSnapshot, formatBytes, formatHardwareSnapshot, modelSafetyHint } from "./hardware.js";
export type { HardwareSnapshot } from "./hardware.js";

// Model artifact cache management
export {
  deleteOneModelArtifactsFromBrowserStorage,
  deleteTransformersModelArtifactsFromBrowserStorage,
  deleteTransformersModelFromBrowserCache,
  deleteWebLLMModelArtifactsFromBrowserStorage,
  estimateOriginStorage,
  getCachedModelStatus,
  hasTransformersModelInBrowserCache,
  matchesModelCacheTarget,
  TRANSFORMERS_CACHE_SCOPE,
  WEBLLM_CACHE_SCOPES,
} from "./model-cache.js";
export type {
  CacheCleanupResult,
  CacheCleanupScopeResult,
  CachedModelStatus,
  IndividualModelCacheStatus,
  IndividualModelCleanupResult,
  ModelCacheTarget,
  StorageEstimateSnapshot,
} from "./model-cache.js";

// Metrics + run history
export { buildMultimodalStats, runRecordFromStats, wordCountGenStats } from "./metrics.js";
export {
  aggregateByModel,
  DEFAULT_HISTORY_KEY,
  DEFAULT_HISTORY_MAX_RECORDS,
  RunHistoryStore,
  VENDOR_LABELS,
  vendorForModel,
} from "./history.js";
export type { ModelAggregate, ModelVendor, RunCaseKey, RunHistoryOptions, RunRecord } from "./history.js";

// Audio utilities
export { AUDIO_SR, encodeWavBlob, pitchShiftBuffer } from "./audio/pcm.js";
export { decodeAudioTo16kMono } from "./audio/decode.js";
export { MicRecorder } from "./audio/recorder.js";
export type { MicRecorderOptions } from "./audio/recorder.js";
export { createStreamingAudioPlayer } from "./audio/streaming-player.js";
export type { StreamingAudioPlayer, StreamingAudioPlayerOptions } from "./audio/streaming-player.js";
export { clipFromRawAudio, clipPlaybackRate, playClipThroughElement, releaseClip, repitchClip, synthesizeClip } from "./audio/clip.js";
export type { SynthClip, SynthesizeClipOptions } from "./audio/clip.js";

// Text utilities
export { stripThinkTags } from "./text-utils.js";

// Voice pipeline
export {
  capPipelineHistory,
  missingPipelineModels,
  PIPELINE_HISTORY_MAX_TURNS,
  PIPELINE_RECOMMENDED,
  PIPELINE_SYSTEM_PROMPT,
  PIPELINE_TASKS,
  shouldSpeakReply,
  streamedDelta,
} from "./voice/helpers.js";
export type { PipelineTask } from "./voice/helpers.js";
export { VoicePipeline } from "./voice/voice-pipeline.js";
export type { TurnDiscardReason, VoicePipelineEvents, VoicePipelineOptions, VoicePipelineStage } from "./voice/voice-pipeline.js";
