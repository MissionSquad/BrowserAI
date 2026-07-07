import type { AppConfig, ModelRecord } from "@mlc-ai/web-llm";
import { applyCuratedModelRecordOverrides, CUSTOM_MODEL_RECORDS } from "./models.js";

type WebLLMModule = typeof import("@mlc-ai/web-llm");

const HUGGING_FACE_HOST = "huggingface.co";
const RAW_GITHUB_HOST = "raw.githubusercontent.com";

export type ModelSourceMode = "direct" | "same-origin-proxy";
export type BrowserCacheBackend = "cache" | "indexeddb" | "opfs" | "cross-origin";

/** The Cache API bucket Transformers.js artifacts are cached in (shared with model-cache cleanup). */
export const TRANSFORMERS_CACHE_KEY = "transformers-cache";

/**
 * Build the WebLLM AppConfig for a load: the bundled prebuilt records merged with the SDK's custom
 * MLC records and curated overrides, optionally rewritten to same-origin proxy routes, with the
 * chosen artifact cache backend applied. `proxyOrigin` is only read in proxy mode.
 */
export function buildWebLLMAppConfig(
  webllm: WebLLMModule,
  mode: ModelSourceMode,
  proxyOrigin: string,
  cacheBackend: BrowserCacheBackend,
): AppConfig {
  const mergedConfig = withCustomModelRecords(webllm.prebuiltAppConfig);
  const baseConfig = mode === "direct" ? mergedConfig : buildProxiedAppConfig(mergedConfig, proxyOrigin);

  return {
    ...baseConfig,
    cacheBackend,
  };
}

export function withCustomModelRecords(appConfig: AppConfig): AppConfig {
  const existingModelIds = new Set(appConfig.model_list.map((record) => record.model_id));
  const missingCustomRecords = CUSTOM_MODEL_RECORDS.filter((record) => !existingModelIds.has(record.model_id));
  const modelList = [...appConfig.model_list, ...missingCustomRecords].map(applyCuratedModelRecordOverrides);

  return {
    ...appConfig,
    model_list: modelList,
  };
}

export function buildProxiedAppConfig(appConfig: AppConfig, origin: string): AppConfig {
  const modelList = appConfig.model_list.map((record) => proxyModelRecord(record, origin));
  return {
    ...appConfig,
    model_list: modelList,
  };
}

export function proxyModelRecord(record: ModelRecord, origin: string): ModelRecord {
  return {
    ...record,
    model: proxyHuggingFaceUrl(record.model, origin),
    // WebLLM's bundled model_lib values usually point at raw.githubusercontent.com, while custom
    // MLC builds may point at Hugging Face. Proxy both forms so hosted Worker deployments stay
    // same-origin and avoid CORS/cache differences.
    model_lib: proxyModelLibraryUrl(record.model_lib, origin),
  };
}

export function proxyModelLibraryUrl(rawUrl: string, origin: string): string {
  return proxyRawGithubUrl(proxyHuggingFaceUrl(rawUrl, origin), origin);
}

export function proxyHuggingFaceUrl(rawUrl: string, origin: string): string {
  let modelUrl: URL;
  try {
    modelUrl = new URL(rawUrl);
  } catch {
    return rawUrl;
  }

  if (modelUrl.hostname !== HUGGING_FACE_HOST) {
    return rawUrl;
  }

  const parts = modelUrl.pathname.split("/").filter(Boolean);
  const [owner, repo, ...suffixParts] = parts;
  if (!owner || !repo) {
    return rawUrl;
  }

  const proxiedPath = ["hf", owner, repo, ...suffixParts].map(encodeURIComponent).join("/");
  const proxied = new URL(`/${proxiedPath}`, origin);
  proxied.search = modelUrl.search;

  return proxied.href;
}

export function proxyRawGithubUrl(rawUrl: string, origin: string): string {
  let modelLibUrl: URL;
  try {
    modelLibUrl = new URL(rawUrl);
  } catch {
    return rawUrl;
  }

  if (modelLibUrl.hostname !== RAW_GITHUB_HOST) {
    return rawUrl;
  }

  const parts = modelLibUrl.pathname.split("/").filter(Boolean);
  const [owner, repo, branch, ...fileParts] = parts;
  if (!owner || !repo || !branch || fileParts.length === 0) {
    return rawUrl;
  }

  const proxiedPath = ["gh-raw", owner, repo, branch, ...fileParts].map(encodeURIComponent).join("/");
  const proxied = new URL(`/${proxiedPath}`, origin);
  proxied.search = modelLibUrl.search;

  return proxied.href;
}

/**
 * Merge a caller-supplied context length into the selected model's record overrides. Preserves any
 * curated overrides (e.g. Gemma's fixed sliding-window config) by merging rather than replacing.
 */
export function applyContextOverride(appConfig: AppConfig, modelId: string, contextLength: number | null): AppConfig {
  if (contextLength === null) return appConfig;
  return {
    ...appConfig,
    model_list: appConfig.model_list.map((record) =>
      record.model_id === modelId ? { ...record, overrides: { ...(record.overrides ?? {}), context_window_size: contextLength } } : record,
    ),
  };
}

/** Human-readable description of where artifacts come from and how they are cached. */
export function modelSourceDescription(mode: ModelSourceMode, cacheBackend: BrowserCacheBackend): string {
  const cacheText = cacheBackend === "indexeddb" ? "browser IndexedDB" : `the browser ${cacheBackend} backend`;

  if (mode === "same-origin-proxy") {
    return `Model files are fetched through the proxy Worker at /hf/* and WebLLM libraries through /gh-raw/*, then cached in ${cacheText}.`;
  }

  return `Model files are fetched directly from Hugging Face and cached in ${cacheText}.`;
}
