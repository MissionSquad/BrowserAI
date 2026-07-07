import type { BrowserCacheBackend } from "./model-source.js";
import type { KokoroModule } from "./multimodal.js";

/** Default WebLLM context-window baseline; a differing override becomes `context_window_size`. */
export const DEFAULT_CONTEXT_LENGTH = 4096;

/** Storage adapter for run-history persistence (matches the `localStorage` surface the SDK uses). */
export type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export type WebLLMOptions = {
  /**
   * Factory for a dedicated module worker running the SDK's WebLLM handler (the
   * `@missionsquad/browserai/worker` entry). When provided, engines are created with
   * `CreateWebWorkerMLCEngine` so loading + token generation stay off the main thread; when absent,
   * the SDK uses the main-thread `CreateMLCEngine`.
   */
  worker?: () => Worker;
  /** WebLLM engine log level. Default "INFO". */
  logLevel?: "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR" | "SILENT";
};

export type KokoroOptions = {
  /** Override the kokoro-js ESM bundle URL (must export KokoroTTS + TextSplitterStream). */
  url?: string;
  /** Fully replace module loading (e.g. a bundled import). Wins over `url`. */
  load?: () => Promise<KokoroModule>;
};

export type TtsDefaults = {
  /** Voice id for the loaded TTS model's roster (Kokoro: "af_heart" etc.; Supertonic: "F1"–"M5"). */
  voice: string;
  /** Speech speed multiplier. Kokoro bakes it into synthesis; Supertonic applies it at playback. */
  speed: number;
  /** Pitch ratio baked into produced clips (1 = unchanged, 2 = up an octave). */
  pitch: number;
};

export type HistoryOptions = {
  /**
   * Persistence for the run log. Defaults to `globalThis.localStorage` when available; pass `null`
   * to keep history in memory only.
   */
  storage?: StorageLike | null;
  /** localStorage key. */
  key?: string;
  /** Cap on persisted records (newest first). */
  maxRecords?: number;
};

export type BrowserAIConfig = {
  /**
   * Where model artifacts are fetched from. "direct" (default) downloads straight from Hugging Face
   * and raw GitHub — works on any site. "proxy" routes through a same-origin (or configured-origin)
   * deployment of the bundled Cloudflare Worker template (`/hf/*`, `/hf-transformers/*`, `/gh-raw/*`).
   */
  modelSource?: "direct" | "proxy";
  /**
   * Origin of the proxy Worker when `modelSource: "proxy"`. Defaults to the page's own origin.
   * Ignored in direct mode.
   */
  proxyOrigin?: string;
  /** Probe the proxy (health + config HEAD checks) before each proxied load. Default true. */
  verifyProxy?: boolean;
  /**
   * WebLLM artifact cache backend. Default "indexeddb" (clearer failures than the Cache API on
   * proxied deployments). Transformers.js artifacts always use the browser Cache API bucket
   * "transformers-cache" regardless of this setting.
   */
  cacheBackend?: BrowserCacheBackend;
  webllm?: WebLLMOptions;
  kokoro?: KokoroOptions;
  /** Default TTS voice/speed/pitch used by synthesize helpers and the voice pipeline. */
  tts?: Partial<TtsDefaults>;
  history?: HistoryOptions;
};

export type ResolvedConfig = {
  modelSource: "direct" | "proxy";
  proxyOrigin: string | null;
  verifyProxy: boolean;
  cacheBackend: BrowserCacheBackend;
  webllm: WebLLMOptions;
  kokoro: KokoroOptions;
  tts: TtsDefaults;
  history: HistoryOptions;
};

/** Apply defaults. `proxyOrigin` falls back to the page origin only when running in a window/worker. */
export function resolveConfig(config: BrowserAIConfig = {}): ResolvedConfig {
  const pageOrigin =
    typeof globalThis.location !== "undefined" && typeof globalThis.location.origin === "string"
      ? globalThis.location.origin
      : null;
  return {
    modelSource: config.modelSource ?? "direct",
    proxyOrigin: config.proxyOrigin ?? pageOrigin,
    verifyProxy: config.verifyProxy ?? true,
    cacheBackend: config.cacheBackend ?? "indexeddb",
    webllm: config.webllm ?? {},
    kokoro: config.kokoro ?? {},
    tts: { voice: "af_heart", speed: 1, pitch: 1, ...config.tts },
    history: config.history ?? {},
  };
}
