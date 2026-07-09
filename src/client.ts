import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import type { ModelPreset, ModelTask } from "./models.js";
import { getModelPreset, MODEL_PRESETS, modelRepoLabel, modelSlots } from "./models.js";
import type {
  ChatMessage,
  EngineHandles,
  GenerationResult,
  RuntimeStats,
  TransformersModule,
  TransformersTextGenerationPipeline,
  TransformersTokenizer,
} from "./inference.js";
import { EMPTY_RESPONSE, generate } from "./inference.js";
import type { RuntimeParameters } from "./generation.js";
import { clampNumber, runtimeParamsFromPreset } from "./generation.js";
import type { MultimodalRuntime } from "./models.js";
import type {
  MultimodalDevice,
  MultimodalHandle,
  MultimodalTransformers,
  RawAudio,
  SttResult,
  TtsTextSplitter,
  VlmResult,
} from "./multimodal.js";
import {
  describeImage as describeImageWithHandle,
  disposeMultimodal,
  generateGemmaText,
  loadMultimodal,
  supportsStreamingTts,
  synthesize as synthesizeWithHandle,
  synthesizeStream,
  transcribe as transcribeWithHandle,
} from "./multimodal.js";
import type { BrowserAIConfig, ResolvedConfig } from "./config.js";
import { DEFAULT_CONTEXT_LENGTH, resolveConfig } from "./config.js";
import type { ModelSourceMode } from "./model-source.js";
import { applyContextOverride, buildWebLLMAppConfig, TRANSFORMERS_CACHE_KEY } from "./model-source.js";
import { verifyTransformersProxy, verifyWebLLMProxy } from "./proxy-verify.js";
import type { HardwareSnapshot } from "./hardware.js";
import { collectHardwareSnapshot, isChromiumBased } from "./hardware.js";
import type { CacheCleanupResult, CachedModelStatus, IndividualModelCleanupResult, ModelCacheTarget, StorageEstimateSnapshot } from "./model-cache.js";
import {
  deleteOneModelArtifactsFromBrowserStorage,
  deleteTransformersModelArtifactsFromBrowserStorage,
  deleteWebLLMModelArtifactsFromBrowserStorage,
  estimateOriginStorage,
  getCachedModelStatus,
} from "./model-cache.js";
import { buildMultimodalStats, runRecordFromStats } from "./metrics.js";
import type { RunRecord } from "./history.js";
import { RunHistoryStore } from "./history.js";
import { TypedEmitter } from "./events.js";
import { BrowserAIError, MissingModelError, ModelLoadError, UnknownModelError, WebGPUUnavailableError } from "./errors.js";
import type { SynthClip, SynthesizeClipOptions } from "./audio/clip.js";
import { synthesizeClip } from "./audio/clip.js";

/**
 * One loaded model and its engine handles. Exactly one backend group is populated: WebLLM
 * (`engine` + `worker`), a Transformers.js text LM (`transformersGenerator`/`Tokenizer`/`Module`),
 * or a multimodal handle (`multimodal` + `mmModule`, for STT/TTS/VLM/Gemma). `slots` are the
 * capability slots it occupies ({@link modelSlots}).
 */
export type LoadedModel = {
  modelId: string;
  slots: ModelTask[];
  backend: "webllm" | "transformers-js";
  engine: MLCEngineInterface | null;
  worker: Worker | null;
  transformersGenerator: TransformersTextGenerationPipeline | null;
  transformersTokenizer: TransformersTokenizer | null;
  transformersModule: TransformersModule | null;
  multimodal: MultimodalHandle | null;
  mmModule: MultimodalTransformers | null;
};

export type LoadProgress = { modelId: string; progress: number; status?: string; file?: string };

/**
 * Multimodal runtimes whose loaders honor a wasm device. The VLM/Gemma/audio-LLM recipes hardcode
 * WebGPU (their verified recipes only run there), so they cannot fall back.
 */
const WASM_CAPABLE_RUNTIMES: ReadonlySet<MultimodalRuntime> = new Set(["stt", "tts-pipeline", "tts-kokoro"]);

export type BrowserAIEvents = {
  /** Download/initialization progress for the model currently loading (progress 0–1). */
  loadprogress: LoadProgress;
  /** Human-readable engine status lines (loading, loaded-in-Ns, proxy checks…). */
  status: { message: string };
  modelloaded: { model: LoadedModel };
  modelunloaded: { modelId: string };
};

export type LoadOptions = {
  /**
   * WebLLM-only context-window override. Ignored when ≤0 or equal to the model's tested default
   * ({@link DEFAULT_CONTEXT_LENGTH}); otherwise clamped to [512, 32768] and merged into the model
   * record's overrides. Transformers.js and multimodal loads ignore it.
   */
  contextLength?: number | null;
  /** Per-call progress callback (also emitted as the "loadprogress" event). */
  onProgress?: (info: LoadProgress) => void;
};

export type GenerateTextOptions = {
  /** Overrides merged over the loaded model's preset defaults. */
  runtime?: Partial<RuntimeParameters>;
  /** JSON Schema for WebLLM constrained decoding (prompt-only backends receive it via the prompt). */
  schema?: unknown;
  /** Stream callback — receives the FULL accumulated text so far on each delta. */
  onDelta?: (fullText: string) => void;
  /** Reset the WebLLM KV cache before generating (default true). */
  resetChat?: boolean;
};

export type TtsOverrides = { voice?: string; speed?: number; pitch?: number };

export type StreamingTtsSession = {
  /** Push text incrementally (e.g. LLM deltas); sentences are synthesized as they complete. */
  splitter: TtsTextSplitter;
  /** One RawAudio per synthesized sentence, in order. Pull-driven. */
  chunks: AsyncIterable<RawAudio>;
  /**
   * MUST be called when the session is finished (after the chunks iterator completes or is
   * abandoned) — it releases the TTS model's serialization lock so other speech work can run.
   */
  release: () => void;
};

/**
 * The BrowserAI client: model catalog + capability-slot manager + inference façade.
 *
 * Models load into one of four capability slots — text / vision / stt / tts — with at most one
 * model per slot (up to four resident at once). Loading a model whose slots conflict with loaded
 * models replaces the conflicting owners. Loads/unloads are single-flight, and runs are serialized
 * per loaded model, so concurrent calls queue instead of driving one engine from two places.
 */
export class BrowserAI extends TypedEmitter<BrowserAIEvents> {
  readonly config: ResolvedConfig;
  #loaded: LoadedModel[] = [];
  #lifecycleBusy = false;
  #runChains = new WeakMap<LoadedModel, Promise<unknown>>();
  #runsInFlight = 0;
  #hardware: HardwareSnapshot | null = null;
  #history: RunHistoryStore | null = null;

  constructor(config: BrowserAIConfig = {}) {
    super();
    this.config = resolveConfig(config);
  }

  /* ------------------------------------------------------------------ */
  /* Catalog + slots                                                     */
  /* ------------------------------------------------------------------ */

  /** The full model catalog. */
  get presets(): readonly ModelPreset[] {
    return MODEL_PRESETS;
  }

  /** Catalog lookup without fallback. */
  getPreset(modelId: string): ModelPreset | undefined {
    return MODEL_PRESETS.find((preset) => preset.id === modelId);
  }

  /** Currently loaded models (read-only snapshot). */
  get loadedModels(): readonly LoadedModel[] {
    return this.#loaded;
  }

  /** The loaded model occupying `task`, if any (≤1 owner per slot — enforced at load). */
  slotOwner(task: ModelTask): LoadedModel | undefined {
    return this.#loaded.find((model) => model.slots.includes(task));
  }

  textModel(): LoadedModel | undefined {
    return this.slotOwner("text");
  }
  visionModel(): LoadedModel | undefined {
    return this.slotOwner("vision");
  }
  sttModel(): LoadedModel | undefined {
    return this.slotOwner("stt");
  }
  ttsModel(): LoadedModel | undefined {
    return this.slotOwner("tts");
  }

  /** Every capability slot currently filled by a loaded model. */
  occupiedSlots(): Set<ModelTask> {
    const filled = new Set<ModelTask>();
    for (const model of this.#loaded) {
      for (const slot of model.slots) filled.add(slot);
    }
    return filled;
  }

  /** Whether `preset` can load without evicting anything (all its slots are free). */
  isLoadable(preset: ModelPreset): boolean {
    const occupied = this.occupiedSlots();
    return modelSlots(preset).every((slot) => !occupied.has(slot));
  }

  /** True while a load/unload/delete is in flight or any generation run is executing. */
  isBusy(): boolean {
    return this.#lifecycleBusy || this.#runsInFlight > 0;
  }

  /** True while a load/unload/cache-delete lifecycle operation is in flight. */
  get loading(): boolean {
    return this.#lifecycleBusy;
  }

  /** The most recent hardware probe (from the last load), if any. */
  get lastHardwareSnapshot(): HardwareSnapshot | null {
    return this.#hardware;
  }

  /** Probe WebGPU + hardware now (also refreshes {@link lastHardwareSnapshot}). */
  async probeHardware(): Promise<HardwareSnapshot> {
    this.#hardware = await collectHardwareSnapshot();
    return this.#hardware;
  }

  /** The run log configured by `config.history` (created lazily; localStorage-backed by default). */
  get history(): RunHistoryStore {
    this.#history ??= new RunHistoryStore(this.config.history);
    return this.#history;
  }

  /**
   * Map a run's stats to a {@link RunRecord} and append it to {@link history}. Returns the record,
   * or null for stats without a model (error placeholders are never logged).
   */
  logRun(caseKey: string, stats: RuntimeStats): RunRecord | null {
    const record = runRecordFromStats(caseKey, stats);
    if (record) this.history.append(record);
    return record;
  }

  /* ------------------------------------------------------------------ */
  /* Load / unload                                                       */
  /* ------------------------------------------------------------------ */

  /**
   * Load a model by catalog id. Evicts any loaded models occupying the slots this one needs
   * (whole-model eviction: a multi-slot owner is fully released even for a one-slot conflict).
   * Single-flight: a second load while one is in flight throws.
   *
   * @throws {UnknownModelError} for an id not in the catalog.
   * @throws {WebGPUUnavailableError} when WebGPU is required and unavailable. Only ASR-pipeline and
   *   TTS presets fall back to wasm; text, vision, Gemma and audio-LLM presets require WebGPU.
   * @throws {ModelLoadError} wrapping any backend load failure.
   */
  async load(modelId: string, options: LoadOptions = {}): Promise<LoadedModel> {
    const preset = this.getPreset(modelId);
    if (!preset) throw new UnknownModelError(modelId);
    if (this.#lifecycleBusy) throw new BrowserAIError("Another load/unload is already in progress.");
    this.#lifecycleBusy = true; // claimed synchronously before any await (protects ≤1-per-slot)
    try {
      this.#hardware = await collectHardwareSnapshot();
      let device: MultimodalDevice = "webgpu";
      const wasmCapable = !!preset.mmRuntime && WASM_CAPABLE_RUNTIMES.has(preset.mmRuntime);
      if (!this.#hardware.webgpuSupported) {
        // Only the runtimes whose loaders honor a wasm device (ASR pipelines, TTS) can fall back;
        // VLM/Gemma/audio-LLM recipes hardcode WebGPU, and text engines require it outright.
        if (wasmCapable) device = "wasm";
        else throw new WebGPUUnavailableError(this.#hardware.webgpuReason);
      } else if (wasmCapable && !isChromiumBased()) {
        // Firefox and Safari now pass the WebGPU probe, but onnxruntime-web's WebGPU backend only
        // reliably runs on Chromium (shader miscompiles / quantized-session failures elsewhere).
        device = "wasm";
      }

      // Free any loaded models occupying the slots this one needs, waiting out their in-flight runs.
      const wanted = modelSlots(preset);
      const conflicts = this.#loaded.filter((model) => model.slots.some((slot) => wanted.includes(slot)));
      for (const conflict of conflicts) {
        await this.#runExclusive(conflict, () => this.#releaseModel(conflict));
      }

      const started = performance.now();
      const progress = (info: Omit<LoadProgress, "modelId">): void => {
        const payload = { modelId: preset.id, ...info };
        this.emit("loadprogress", payload);
        options.onProgress?.(payload);
      };

      let model: LoadedModel;
      try {
        if (preset.mmRuntime) {
          model = await this.#loadMultimodalModel(preset, device, progress);
        } else if (preset.backend === "transformers-js") {
          model = await this.#loadTransformersModel(preset, progress);
        } else {
          model = await this.#loadWebLLMModel(preset, options, progress);
        }
      } catch (error) {
        throw error instanceof BrowserAIError ? error : new ModelLoadError(preset.id, error);
      }

      this.#loaded.push(model);
      this.#status(`Loaded ${preset.shortName} in ${((performance.now() - started) / 1000).toFixed(1)}s.`);
      this.emit("modelloaded", { model });
      return model;
    } finally {
      this.#lifecycleBusy = false;
    }
  }

  /** Unload one model, waiting out its in-flight runs. No-op when the id is not loaded. */
  async unload(modelId: string): Promise<void> {
    const model = this.#loaded.find((entry) => entry.modelId === modelId);
    if (!model) return;
    if (this.#lifecycleBusy) throw new BrowserAIError("Another load/unload is already in progress.");
    this.#lifecycleBusy = true;
    try {
      await this.#runExclusive(model, () => this.#releaseModel(model));
    } finally {
      this.#lifecycleBusy = false;
    }
  }

  /** Unload every loaded model. */
  async unloadAll(): Promise<void> {
    if (this.#lifecycleBusy) throw new BrowserAIError("Another load/unload is already in progress.");
    this.#lifecycleBusy = true;
    try {
      await this.#releaseAll();
    } finally {
      this.#lifecycleBusy = false;
    }
  }

  async #releaseAll(): Promise<void> {
    for (const model of [...this.#loaded]) {
      await this.#runExclusive(model, () => this.#releaseModel(model));
    }
  }

  /** Tear an engine down: WebLLM unload, Transformers dispose, multimodal dispose, worker terminate. */
  async #releaseModel(model: LoadedModel): Promise<void> {
    try {
      try {
        await model.engine?.unload();
      } catch {
        // Best-effort: unload can reject after WebGPU device loss / a crashed worker. The worker
        // still gets terminated below and the tab releases the GPU context regardless.
      }
      await disposeTransformersGenerator(model.transformersGenerator);
      await disposeMultimodal(model.multimodal);
    } finally {
      model.worker?.terminate(); // never leak the worker, even when unload rejected
      this.#loaded = this.#loaded.filter((entry) => entry !== model); // always drop from the slot table
      this.emit("modelunloaded", { modelId: model.modelId });
    }
  }

  /* ------------------------------------------------------------------ */
  /* Backend loaders                                                     */
  /* ------------------------------------------------------------------ */

  #modelSourceMode(): ModelSourceMode {
    return this.config.modelSource === "proxy" ? "same-origin-proxy" : "direct";
  }

  #requireProxyOrigin(): string {
    const origin = this.config.proxyOrigin;
    if (!origin) {
      throw new BrowserAIError('modelSource is "proxy" but no proxyOrigin is configured and no page origin is available.');
    }
    return origin;
  }

  async #loadWebLLMModel(preset: ModelPreset, options: LoadOptions, progress: (info: Omit<LoadProgress, "modelId">) => void): Promise<LoadedModel> {
    this.#status(`Loading ${preset.shortName}… First load downloads and caches the model, so it can take a while.`);

    const webllm = await import("@mlc-ai/web-llm");
    const mode = this.#modelSourceMode();
    const proxyOrigin = mode === "same-origin-proxy" ? this.#requireProxyOrigin() : "https://invalid.example";
    let appConfig = buildWebLLMAppConfig(webllm, mode, proxyOrigin, this.config.cacheBackend);
    appConfig = applyContextOverride(appConfig, preset.id, normalizeContextLength(options.contextLength));

    if (mode === "same-origin-proxy" && this.config.verifyProxy) {
      this.#status("Checking model proxy…");
      await verifyWebLLMProxy(proxyOrigin, preset.id, appConfig);
    }

    const engineConfig = {
      appConfig,
      logLevel: this.config.webllm.logLevel ?? "INFO",
      initProgressCallback: (report: { progress: number; text: string }) => {
        progress({ progress: Number.isFinite(report.progress) ? report.progress : 0, status: report.text || `Loading ${preset.shortName}…` });
        this.#status(report.text || `Loading ${preset.shortName}…`);
      },
    };

    let engine: MLCEngineInterface;
    let worker: Worker | null = null;
    if (this.config.webllm.worker) {
      worker = this.config.webllm.worker();
      try {
        engine = await webllm.CreateWebWorkerMLCEngine(worker, preset.id, engineConfig);
      } catch (error) {
        worker.terminate(); // don't leak the worker if engine creation fails
        throw error;
      }
    } else {
      engine = await webllm.CreateMLCEngine(preset.id, engineConfig);
    }
    progress({ progress: 1 });

    return { ...newLoadedModel(preset), engine, worker };
  }

  async #loadTransformersModel(preset: ModelPreset, progress: (info: Omit<LoadProgress, "modelId">) => void): Promise<LoadedModel> {
    this.#status(`Loading ${preset.shortName} with Transformers.js/WebGPU…`);
    progress({ progress: 0.05 });

    const transformers = (await import("@huggingface/transformers")) as unknown as TransformersModule;
    this.#configureTransformersEnvironment(transformers);

    if (this.#modelSourceMode() === "same-origin-proxy" && this.config.verifyProxy) {
      this.#status("Checking Transformers.js model proxy…");
      await verifyTransformersProxy(this.#requireProxyOrigin(), preset.id);
    }

    const generator = await transformers.pipeline("text-generation", preset.id, {
      dtype: preset.transformersDtype ?? "q4",
      device: "webgpu",
      progress_callback: this.#makeTransformersProgressForwarder(progress),
    });

    progress({ progress: 1 });
    return {
      ...newLoadedModel(preset),
      transformersGenerator: generator,
      transformersTokenizer: generator.tokenizer ?? null,
      transformersModule: transformers,
    };
  }

  async #loadMultimodalModel(preset: ModelPreset, device: MultimodalDevice, progress: (info: Omit<LoadProgress, "modelId">) => void): Promise<LoadedModel> {
    this.#status(`Loading ${preset.shortName} with Transformers.js/WebGPU…`);
    progress({ progress: 0.05 });

    const transformers = (await import("@huggingface/transformers")) as unknown as MultimodalTransformers;
    this.#configureTransformersEnvironment(transformers as unknown as TransformersModule);

    // Kokoro (kokoro-js) bundles its own runtime and fetches weights directly; others use the proxy.
    if (this.#modelSourceMode() === "same-origin-proxy" && this.config.verifyProxy && preset.mmRuntime !== "tts-kokoro") {
      this.#status("Checking Transformers.js model proxy…");
      await verifyTransformersProxy(this.#requireProxyOrigin(), preset.id);
    }

    const handle = await loadMultimodal(preset, transformers, device, this.#makeTransformersProgressForwarder(progress), {
      kokoro: this.config.kokoro,
    });

    progress({ progress: 1 });
    return { ...newLoadedModel(preset), multimodal: handle, mmModule: transformers };
  }

  /**
   * Adapt Transformers.js progress callbacks (percentages 0–100 interleaved with status-only
   * initiate/download/done events) to the SDK's 0–1 loadprogress. Status-only events re-emit the
   * last numeric value instead of snapping the reported progress back to zero.
   */
  #makeTransformersProgressForwarder(progress: (info: Omit<LoadProgress, "modelId">) => void): (info: unknown) => void {
    let lastProgress = 0;
    return (info: unknown) => {
      const report = info as { progress?: number; status?: string; file?: string };
      const label = [report.status, report.file].filter(Boolean).join(" ");
      if (typeof report.progress === "number") {
        lastProgress = Math.max(0, Math.min(1, report.progress / 100));
        progress({ progress: lastProgress, status: report.status, file: report.file });
      } else if (label) {
        progress({ progress: lastProgress, status: report.status, file: report.file });
      }
      if (label) this.#status(label);
    };
  }

  #configureTransformersEnvironment(transformers: TransformersModule): void {
    const env = transformers.env;
    if (!env) return;
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;
    env.useFSCache = false;
    env.cacheKey = TRANSFORMERS_CACHE_KEY;
    if (this.#modelSourceMode() === "same-origin-proxy") {
      env.remoteHost = `${this.#requireProxyOrigin()}/hf-transformers/`;
      env.remotePathTemplate = "{model}/resolve/{revision}/";
    } else {
      env.remoteHost = "https://huggingface.co/";
      env.remotePathTemplate = "{model}/resolve/{revision}/";
    }
  }

  /* ------------------------------------------------------------------ */
  /* Inference façade                                                    */
  /* ------------------------------------------------------------------ */

  /**
   * Run one text generation against the loaded Text-slot model. A Gemma-4 multimodal handle has no
   * WebLLM engine or Transformers.js LM pipeline, so it routes through the Gemma text path;
   * everything else goes to the shared backend-agnostic generator. Runtime parameters default to
   * the loaded model's preset defaults, with `options.runtime` merged on top.
   */
  async generateText(messages: ChatMessage[], options: GenerateTextOptions = {}): Promise<GenerationResult> {
    const model = this.textModel();
    if (!model) throw new MissingModelError("text");
    const preset = getModelPreset(model.modelId);
    const runtime: RuntimeParameters = { ...runtimeParamsFromPreset(preset.defaultRuntime), ...options.runtime };

    return this.#runExclusive(model, async () => {
      const handle = model.multimodal;
      if (handle && handle.kind === "gemma") {
        const gen = await generateGemmaText(handle, messages, {
          maxNewTokens: runtime.maxTokens,
          temperature: runtime.temperature,
          topP: runtime.topP,
          repetitionPenalty: runtime.repetitionPenalty,
          onDelta: options.onDelta,
        });
        const stats: RuntimeStats = buildMultimodalStats(gen, model.modelId);
        // Surface likely truncation the same way the Transformers.js path does.
        stats.finishReason = gen.completionTokens >= runtime.maxTokens ? "length" : "stop";
        return { text: gen.text || EMPTY_RESPONSE, stats };
      }
      return generate(engineHandles(model), messages, runtime, preset, {
        schema: options.schema,
        onDelta: options.onDelta,
        resetChat: options.resetChat,
      });
    });
  }

  /** Reset the WebLLM KV cache of the loaded text model (e.g. for a "new chat"). Queues behind in-flight runs. */
  async resetTextChat(): Promise<void> {
    const model = this.textModel();
    if (!model?.engine) return;
    await this.#runExclusive(model, async () => {
      await model.engine?.resetChat(false);
    });
  }

  /** Transcribe 16 kHz mono PCM (or a URL) with the loaded Transcription-slot model. */
  async transcribe(audio: Float32Array | string, onDelta?: (text: string) => void): Promise<SttResult> {
    const model = this.sttModel();
    if (!model?.multimodal) throw new MissingModelError("stt");
    const handle = model.multimodal;
    return this.#runExclusive(model, () => transcribeWithHandle(handle, audio, onDelta));
  }

  /** Synthesize speech with the loaded Speech-slot model. Returns the raw model audio. */
  async synthesize(text: string, overrides: TtsOverrides = {}): Promise<RawAudio> {
    const model = this.ttsModel();
    if (!model?.multimodal) throw new MissingModelError("tts");
    const handle = model.multimodal;
    const { voice, speed } = { ...this.config.tts, ...overrides };
    return this.#runExclusive(model, () => synthesizeWithHandle(handle, text, { voice, speed }));
  }

  /**
   * Synthesize speech and package it as a playable clip (pitch baked into the blob, original PCM
   * kept for re-pitching). Voice/speed/pitch default to the client's TTS config.
   */
  async synthesizeToClip(text: string, overrides: TtsOverrides & Pick<SynthesizeClipOptions, "createUrl"> = {}): Promise<SynthClip> {
    const model = this.ttsModel();
    if (!model?.multimodal) throw new MissingModelError("tts");
    const handle = model.multimodal;
    const { voice, speed, pitch } = { ...this.config.tts, ...overrides };
    return this.#runExclusive(model, () => synthesizeClip(handle, text, { voice, speed, pitch, createUrl: overrides.createUrl }));
  }

  /** Whether the loaded Speech-slot model supports sentence-streaming synthesis (Kokoro). */
  supportsStreamingTts(): boolean {
    return supportsStreamingTts(this.ttsModel()?.multimodal);
  }

  /**
   * Begin a streaming TTS session on the loaded Kokoro model: push LLM deltas into `splitter`,
   * consume `chunks` (one RawAudio per sentence), then call `release()` when done. The session
   * holds the TTS model's serialization lock until released so no other speech run interleaves —
   * the returned promise resolves only once the lock is actually held (queued behind in-flight
   * speech runs), and the Kokoro stream is created after that point.
   * @throws {MissingModelError} when no Speech model is loaded or it cannot stream.
   */
  async createStreamingTts(overrides: Pick<TtsOverrides, "voice" | "speed"> = {}): Promise<StreamingTtsSession> {
    const model = this.ttsModel();
    const handle = model?.multimodal;
    if (!model || !supportsStreamingTts(handle)) {
      throw new MissingModelError("tts", "Streaming TTS requires a loaded Kokoro speech model.");
    }
    const { voice, speed } = { ...this.config.tts, ...overrides };

    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    // Occupy the model's run chain for the session's lifetime (released by release()), and wait
    // until the chain actually reaches this session before touching the engine.
    const acquired = new Promise<void>((resolve) => {
      void this.#runExclusive(model, () => {
        resolve();
        return gate;
      });
    });
    await acquired;

    const { splitter, chunks } = synthesizeStream(handle, { voice, speed });

    let released = false;
    return {
      splitter,
      chunks,
      release: () => {
        if (released) return;
        released = true;
        release();
      },
    };
  }

  /** Ask a question about an image with the loaded Image-slot model (VLM or Gemma). */
  async describeImage(imageSrc: string, prompt: string, onDelta?: (text: string) => void): Promise<VlmResult> {
    const model = this.visionModel();
    if (!model?.multimodal) throw new MissingModelError("vision");
    const handle = model.multimodal;
    return this.#runExclusive(model, () => describeImageWithHandle(handle, imageSrc, prompt, onDelta));
  }

  /* ------------------------------------------------------------------ */
  /* Cache management                                                    */
  /* ------------------------------------------------------------------ */

  /** Whether a model's artifacts are present in browser storage (Cache API / IndexedDB). */
  async cacheStatus(modelId: string): Promise<CachedModelStatus> {
    const preset = this.getPreset(modelId);
    if (!preset) throw new UnknownModelError(modelId);
    return getCachedModelStatus(buildModelCacheTarget(preset));
  }

  /** Origin-wide storage usage/quota estimate (browser-rounded). */
  estimateStorage(): Promise<StorageEstimateSnapshot | undefined> {
    return estimateOriginStorage();
  }

  /**
   * Delete one model's downloaded artifacts. If the model is loaded it is released first (its
   * in-flight runs are awaited). Targets model artifact storage only.
   */
  async deleteModelArtifacts(modelId: string): Promise<IndividualModelCleanupResult> {
    const preset = this.getPreset(modelId);
    if (!preset) throw new UnknownModelError(modelId);
    if (this.#lifecycleBusy) throw new BrowserAIError("Another load/unload is already in progress.");
    this.#lifecycleBusy = true;
    try {
      const loaded = this.#loaded.find((entry) => entry.modelId === modelId);
      if (loaded) await this.#runExclusive(loaded, () => this.#releaseModel(loaded));
      return await deleteOneModelArtifactsFromBrowserStorage(buildModelCacheTarget(preset));
    } finally {
      this.#lifecycleBusy = false;
    }
  }

  /** Delete ALL downloaded model artifacts (WebLLM + Transformers.js), releasing loaded models first. */
  async deleteAllModelArtifacts(): Promise<CacheCleanupResult> {
    if (this.#lifecycleBusy) throw new BrowserAIError("Another load/unload is already in progress.");
    this.#lifecycleBusy = true;
    try {
      // Bracket BOTH deletions with the storage estimate so freedBytes reflects the Transformers.js
      // cache too (the WebLLM helper's own after-estimate runs before that deletion happens).
      const before = await estimateOriginStorage();
      await this.#releaseAll();
      const webllmResult = await deleteWebLLMModelArtifactsFromBrowserStorage();
      const transformersScope = await deleteTransformersModelArtifactsFromBrowserStorage();
      await new Promise((resolve) => setTimeout(resolve, 250)); // let the estimate settle
      const after = await estimateOriginStorage();
      const freedBytes =
        typeof before?.usage === "number" && typeof after?.usage === "number" ? Math.max(0, before.usage - after.usage) : undefined;
      return { before, after, freedBytes, scopes: [...webllmResult.scopes, transformersScope] };
    } finally {
      this.#lifecycleBusy = false;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Internals                                                           */
  /* ------------------------------------------------------------------ */

  /** Serialize work per loaded model: a run (or teardown) waits for prior runs on the same model. */
  #runExclusive<T>(model: LoadedModel, fn: () => Promise<T>): Promise<T> {
    const prior = this.#runChains.get(model) ?? Promise.resolve();
    const run = prior.then(async () => {
      this.#runsInFlight += 1;
      try {
        return await fn();
      } finally {
        this.#runsInFlight -= 1;
      }
    });
    this.#runChains.set(
      model,
      run.then(
        () => undefined,
        () => undefined,
      ),
    );
    return run;
  }

  #status(message: string): void {
    this.emit("status", { message });
  }
}

/** A blank LoadedModel record for `preset` (the backend-specific loader fills in its runtime handles). */
function newLoadedModel(preset: ModelPreset): LoadedModel {
  return {
    modelId: preset.id,
    slots: modelSlots(preset),
    backend: preset.backend,
    engine: null,
    worker: null,
    transformersGenerator: null,
    transformersTokenizer: null,
    transformersModule: null,
    multimodal: null,
    mmModule: null,
  };
}

function engineHandles(model: LoadedModel): EngineHandles {
  return {
    engine: model.engine,
    transformersGenerator: model.transformersGenerator,
    transformersTokenizer: model.transformersTokenizer,
    transformersModule: model.transformersModule,
    backend: model.backend,
    modelId: model.modelId,
  };
}

/** Best-effort teardown for a Transformers.js text pipeline (API shape varies across versions). */
async function disposeTransformersGenerator(generator: TransformersTextGenerationPipeline | null): Promise<void> {
  if (!generator) return;
  try {
    if (typeof generator.dispose === "function") {
      await generator.dispose();
      return;
    }
    if (generator.model && typeof generator.model.dispose === "function") {
      await generator.model.dispose();
    }
  } catch {
    // Best-effort cleanup; closing the tab releases the WebGPU context regardless.
  }
}

/**
 * Normalize a context-length override: null/undefined, non-positive, or exactly the tested default
 * ({@link DEFAULT_CONTEXT_LENGTH}) mean "no override"; anything else clamps to [512, 32768].
 */
export function normalizeContextLength(value: number | null | undefined): number | null {
  if (value === null || value === undefined) return null;
  if (!Number.isFinite(value) || value <= 0 || value === DEFAULT_CONTEXT_LENGTH) return null;
  return clampNumber(value, 512, 32768);
}

/** The cache-target matchers (id, repo labels, artifact URLs) for a preset's downloaded artifacts. */
export function buildModelCacheTarget(preset: ModelPreset): ModelCacheTarget {
  const repoLabels = new Set<string>([preset.id, modelRepoLabel(preset)]);
  for (const repoUrl of [preset.officialRepo, preset.artifactRepo].filter(Boolean) as string[]) {
    try {
      repoLabels.add(new URL(repoUrl).pathname.replace(/^\//, ""));
    } catch {
      // Ignore malformed optional repo URLs.
    }
  }
  return {
    modelId: preset.id,
    repoLabels: Array.from(repoLabels),
    artifactUrls: [preset.officialRepo, preset.artifactRepo, preset.id].filter(Boolean) as string[],
  };
}
