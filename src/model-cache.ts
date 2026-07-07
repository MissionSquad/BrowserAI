export type StorageEstimateSnapshot = {
  usage?: number;
  quota?: number;
};

export type CacheCleanupScopeResult = {
  kind: "cache-storage" | "indexeddb" | "opfs" | "transformers-cache";
  name: string;
  supported: boolean;
  cleared: boolean;
  error?: string;
};

export type CacheCleanupResult = {
  before?: StorageEstimateSnapshot;
  after?: StorageEstimateSnapshot;
  freedBytes?: number;
  scopes: CacheCleanupScopeResult[];
};

// These are the WebLLM artifact stores used by the SDK's selectable cache backends. Deleting these
// targets model downloads only; it does not call caches.keys()/indexedDB.databases() broadly and
// does not touch localStorage, cookies, or unrelated site data.
export const WEBLLM_CACHE_SCOPES = [
  "webllm/model",
  "webllm/config",
  "webllm/wasm",
] as const;
const WEBLLM_OPFS_ROOT = "tvmjs-opfs-store";

export async function estimateOriginStorage(): Promise<
  StorageEstimateSnapshot | undefined
> {
  try {
    const estimate = await navigator.storage?.estimate?.();
    if (!estimate) return undefined;
    return {
      usage: typeof estimate.usage === "number" ? estimate.usage : undefined,
      quota: typeof estimate.quota === "number" ? estimate.quota : undefined,
    };
  } catch {
    return undefined;
  }
}

export async function deleteWebLLMModelArtifactsFromBrowserStorage(): Promise<CacheCleanupResult> {
  const before = await estimateOriginStorage();
  const scopes: CacheCleanupScopeResult[] = [];

  for (const scope of WEBLLM_CACHE_SCOPES) {
    scopes.push(await deleteCacheStorageScope(scope));
  }

  for (const scope of WEBLLM_CACHE_SCOPES) {
    scopes.push(await deleteIndexedDBDatabase(scope));
  }

  scopes.push(await deleteOPFSWebLLMRoot());

  await new Promise((resolve) => setTimeout(resolve, 250));
  const after = await estimateOriginStorage();
  const freedBytes =
    typeof before?.usage === "number" && typeof after?.usage === "number"
      ? Math.max(0, before.usage - after.usage)
      : undefined;

  return { before, after, freedBytes, scopes };
}

async function deleteCacheStorageScope(
  name: string,
): Promise<CacheCleanupScopeResult> {
  if (!("caches" in globalThis)) {
    return {
      kind: "cache-storage",
      name,
      supported: false,
      cleared: false,
      error: "Cache Storage API is unavailable.",
    };
  }

  try {
    const cleared = await caches.delete(name);
    return { kind: "cache-storage", name, supported: true, cleared };
  } catch (error) {
    return {
      kind: "cache-storage",
      name,
      supported: true,
      cleared: false,
      error: formatUnknownError(error),
    };
  }
}

async function deleteIndexedDBDatabase(
  name: string,
): Promise<CacheCleanupScopeResult> {
  if (!("indexedDB" in globalThis)) {
    return {
      kind: "indexeddb",
      name,
      supported: false,
      cleared: false,
      error: "IndexedDB is unavailable.",
    };
  }

  return new Promise((resolve) => {
    let settled = false;
    let blocked = false;
    const request = indexedDB.deleteDatabase(name);

    const finish = (result: CacheCleanupScopeResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };

    const timer = setTimeout(() => {
      finish({
        kind: "indexeddb",
        name,
        supported: true,
        cleared: false,
        error: blocked
          ? "Deletion is blocked by an open IndexedDB connection. Close other tabs for this site and try again."
          : "IndexedDB deletion timed out. Close other tabs for this site and try again.",
      });
    }, 5000);

    request.onblocked = () => {
      blocked = true;
    };
    request.onsuccess = () => {
      finish({ kind: "indexeddb", name, supported: true, cleared: true });
    };
    request.onerror = () => {
      finish({
        kind: "indexeddb",
        name,
        supported: true,
        cleared: false,
        error: request.error?.message ?? "Unknown IndexedDB error.",
      });
    };
  });
}

async function deleteOPFSWebLLMRoot(): Promise<CacheCleanupScopeResult> {
  type StorageManagerWithDirectory = StorageManager & {
    getDirectory?: () => Promise<{
      removeEntry: (
        name: string,
        options?: { recursive?: boolean },
      ) => Promise<void>;
    }>;
  };

  try {
    const storage = navigator.storage as
      StorageManagerWithDirectory | undefined;
    if (typeof storage?.getDirectory !== "function") {
      return {
        kind: "opfs",
        name: WEBLLM_OPFS_ROOT,
        supported: false,
        cleared: false,
        error: "OPFS is unavailable.",
      };
    }

    const root = await storage.getDirectory();
    await root.removeEntry(WEBLLM_OPFS_ROOT, { recursive: true });
    return {
      kind: "opfs",
      name: WEBLLM_OPFS_ROOT,
      supported: true,
      cleared: true,
    };
  } catch (error) {
    if (error instanceof DOMException && error.name === "NotFoundError") {
      return {
        kind: "opfs",
        name: WEBLLM_OPFS_ROOT,
        supported: true,
        cleared: false,
      };
    }
    return {
      kind: "opfs",
      name: WEBLLM_OPFS_ROOT,
      supported: true,
      cleared: false,
      error: formatUnknownError(error),
    };
  }
}

export async function deleteTransformersModelArtifactsFromBrowserStorage(): Promise<CacheCleanupScopeResult> {
  return deleteCacheStorageScope(TRANSFORMERS_CACHE_SCOPE);
}

function formatUnknownError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

export const TRANSFORMERS_CACHE_SCOPE = "transformers-cache";

export type IndividualModelCacheStatus = {
  supported: boolean;
  downloaded: boolean;
  detail: string;
};

export async function hasTransformersModelInBrowserCache(
  repoId: string,
): Promise<IndividualModelCacheStatus> {
  if (!("caches" in globalThis)) {
    return {
      supported: false,
      downloaded: false,
      detail: "Cache Storage API is unavailable.",
    };
  }

  try {
    const cache = await caches.open(TRANSFORMERS_CACHE_SCOPE);
    const keys = await cache.keys();
    const matchCount = keys.filter((request) =>
      isTransformersModelRequest(request.url, repoId),
    ).length;
    return {
      supported: true,
      downloaded: matchCount > 0,
      detail:
        matchCount > 0
          ? `${matchCount} cached Transformers.js artifacts found.`
          : "No cached Transformers.js artifacts found.",
    };
  } catch (error) {
    return {
      supported: true,
      downloaded: false,
      detail: `Could not inspect Transformers.js cache: ${formatUnknownError(error)}`,
    };
  }
}

export async function deleteTransformersModelFromBrowserCache(
  repoId: string,
): Promise<number> {
  if (!("caches" in globalThis)) return 0;

  const cache = await caches.open(TRANSFORMERS_CACHE_SCOPE);
  const keys = await cache.keys();
  let deleted = 0;
  for (const request of keys) {
    if (
      isTransformersModelRequest(request.url, repoId) &&
      (await cache.delete(request))
    ) {
      deleted += 1;
    }
  }
  return deleted;
}

function isTransformersModelRequest(url: string, repoId: string): boolean {
  const normalizedRepo = repoId.replace(/^\/+|\/+$/g, "");
  const encodedRepo = normalizedRepo
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  return (
    url.includes(`/${normalizedRepo}/resolve/`) ||
    url.includes(`/${encodedRepo}/resolve/`) ||
    url.includes(`/hf-transformers/${normalizedRepo}/resolve/`) ||
    url.includes(`/hf-transformers/${encodedRepo}/resolve/`)
  );
}

export type ModelCacheTarget = {
  modelId: string;
  repoLabels: string[];
  artifactUrls: string[];
};

export type CachedModelStatus = {
  downloaded: boolean;
  matchedEntries: number;
  storageKinds: string[];
  sampleKeys: string[];
};

export type IndividualModelCleanupResult = CacheCleanupResult & {
  targetModelId: string;
  deletedEntries: CacheCleanupScopeResult[];
  matchedBefore: CachedModelStatus;
};

export async function getCachedModelStatus(
  target: ModelCacheTarget,
): Promise<CachedModelStatus> {
  const [cacheKeys, indexedDbKeys, transformerKeys] = await Promise.all([
    listMatchingCacheStorageKeys(target),
    listMatchingIndexedDBKeys(target),
    listMatchingTransformersCacheKeys(target),
  ]);

  const matches = [...cacheKeys, ...indexedDbKeys, ...transformerKeys];
  const storageKinds = Array.from(new Set(matches.map((entry) => entry.kind)));
  return {
    downloaded: matches.length > 0,
    matchedEntries: matches.length,
    storageKinds,
    sampleKeys: matches.slice(0, 5).map((entry) => entry.key),
  };
}

export async function deleteOneModelArtifactsFromBrowserStorage(
  target: ModelCacheTarget,
): Promise<IndividualModelCleanupResult> {
  const before = await estimateOriginStorage();
  const matchedBefore = await getCachedModelStatus(target);
  const deletedEntries: CacheCleanupScopeResult[] = [];

  for (const scope of WEBLLM_CACHE_SCOPES) {
    deletedEntries.push(
      ...(await deleteMatchingCacheStorageKeys(scope, target)),
    );
  }
  for (const scope of WEBLLM_CACHE_SCOPES) {
    deletedEntries.push(...(await deleteMatchingIndexedDBKeys(scope, target)));
  }
  deletedEntries.push(...(await deleteMatchingTransformersCacheKeys(target)));

  await new Promise((resolve) => setTimeout(resolve, 250));
  const after = await estimateOriginStorage();
  const freedBytes =
    typeof before?.usage === "number" && typeof after?.usage === "number"
      ? Math.max(0, before.usage - after.usage)
      : undefined;

  return {
    targetModelId: target.modelId,
    before,
    after,
    freedBytes,
    scopes: deletedEntries,
    deletedEntries,
    matchedBefore,
  };
}

async function listMatchingTransformersCacheKeys(
  target: ModelCacheTarget,
): Promise<MatchingStoredKey[]> {
  if (!("caches" in globalThis)) return [];
  try {
    const cache = await caches.open(TRANSFORMERS_CACHE_SCOPE);
    const requests = await cache.keys();
    return requests
      .filter((request) => matchesModelCacheTarget(request.url, target))
      .map((request) => ({
        kind: "transformers-cache" as const,
        scope: TRANSFORMERS_CACHE_SCOPE,
        key: request.url,
      }));
  } catch {
    return [];
  }
}

async function deleteMatchingTransformersCacheKeys(
  target: ModelCacheTarget,
): Promise<CacheCleanupScopeResult[]> {
  if (!("caches" in globalThis)) {
    return [
      {
        kind: "transformers-cache",
        name: TRANSFORMERS_CACHE_SCOPE,
        supported: false,
        cleared: false,
        error: "Cache Storage API is unavailable.",
      },
    ];
  }

  const results: CacheCleanupScopeResult[] = [];
  try {
    const cache = await caches.open(TRANSFORMERS_CACHE_SCOPE);
    const requests = await cache.keys();
    for (const request of requests) {
      if (!matchesModelCacheTarget(request.url, target)) continue;
      const cleared = await cache.delete(request);
      results.push({
        kind: "transformers-cache",
        name: `${TRANSFORMERS_CACHE_SCOPE}:${request.url}`,
        supported: true,
        cleared,
      });
    }
  } catch (error) {
    results.push({
      kind: "transformers-cache",
      name: TRANSFORMERS_CACHE_SCOPE,
      supported: true,
      cleared: false,
      error: formatUnknownError(error),
    });
  }
  return results;
}

type MatchingStoredKey = {
  kind: "cache-storage" | "indexeddb" | "transformers-cache";
  scope: string;
  key: string;
};

async function listMatchingCacheStorageKeys(
  target: ModelCacheTarget,
): Promise<MatchingStoredKey[]> {
  if (!("caches" in globalThis)) return [];
  const matches: MatchingStoredKey[] = [];
  for (const scope of WEBLLM_CACHE_SCOPES) {
    try {
      const cache = await caches.open(scope);
      const requests = await cache.keys();
      for (const request of requests) {
        if (matchesModelCacheTarget(request.url, target)) {
          matches.push({ kind: "cache-storage", scope, key: request.url });
        }
      }
    } catch {
      // Ignore per-scope read failures in status checks. The delete action reports errors.
    }
  }
  return matches;
}

async function listMatchingIndexedDBKeys(
  target: ModelCacheTarget,
): Promise<MatchingStoredKey[]> {
  if (!("indexedDB" in globalThis)) return [];
  const matches: MatchingStoredKey[] = [];
  for (const scope of WEBLLM_CACHE_SCOPES) {
    try {
      const keys = await listIndexedDBUrlKeys(scope);
      for (const key of keys) {
        if (matchesModelCacheTarget(key, target)) {
          matches.push({ kind: "indexeddb", scope, key });
        }
      }
    } catch {
      // Ignore per-scope read failures in status checks. The delete action reports errors.
    }
  }
  return matches;
}

async function deleteMatchingCacheStorageKeys(
  scope: string,
  target: ModelCacheTarget,
): Promise<CacheCleanupScopeResult[]> {
  if (!("caches" in globalThis)) {
    return [
      {
        kind: "cache-storage",
        name: scope,
        supported: false,
        cleared: false,
        error: "Cache Storage API is unavailable.",
      },
    ];
  }

  const results: CacheCleanupScopeResult[] = [];
  try {
    const cache = await caches.open(scope);
    const requests = await cache.keys();
    for (const request of requests) {
      if (!matchesModelCacheTarget(request.url, target)) continue;
      const cleared = await cache.delete(request);
      results.push({
        kind: "cache-storage",
        name: `${scope}:${request.url}`,
        supported: true,
        cleared,
      });
    }
  } catch (error) {
    results.push({
      kind: "cache-storage",
      name: scope,
      supported: true,
      cleared: false,
      error: formatUnknownError(error),
    });
  }
  return results;
}

async function deleteMatchingIndexedDBKeys(
  scope: string,
  target: ModelCacheTarget,
): Promise<CacheCleanupScopeResult[]> {
  if (!("indexedDB" in globalThis)) {
    return [
      {
        kind: "indexeddb",
        name: scope,
        supported: false,
        cleared: false,
        error: "IndexedDB is unavailable.",
      },
    ];
  }

  const results: CacheCleanupScopeResult[] = [];
  try {
    const keys = await listIndexedDBUrlKeys(scope);
    for (const key of keys) {
      if (!matchesModelCacheTarget(key, target)) continue;
      await deleteIndexedDBUrlKey(scope, key);
      results.push({
        kind: "indexeddb",
        name: `${scope}:${key}`,
        supported: true,
        cleared: true,
      });
    }
  } catch (error) {
    results.push({
      kind: "indexeddb",
      name: scope,
      supported: true,
      cleared: false,
      error: formatUnknownError(error),
    });
  }
  return results;
}

function listIndexedDBUrlKeys(dbName: string): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, 1);
    request.onupgradeneeded = () => {
      // The database did not exist yet. Avoid creating extra object stores for status checks.
      request.transaction?.abort();
      resolve([]);
    };
    request.onerror = () => {
      if (request.error?.name === "AbortError") return;
      reject(request.error ?? new Error(`Could not open ${dbName}`));
    };
    request.onsuccess = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("urls")) {
        db.close();
        resolve([]);
        return;
      }
      const tx = db.transaction("urls", "readonly");
      const store = tx.objectStore("urls");
      const keyRequest = store.getAllKeys();
      keyRequest.onsuccess = () => {
        db.close();
        resolve(keyRequest.result.map((key) => String(key)));
      };
      keyRequest.onerror = () => {
        db.close();
        reject(
          keyRequest.error ?? new Error(`Could not list keys for ${dbName}`),
        );
      };
    };
  });
}

function deleteIndexedDBUrlKey(dbName: string, key: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, 1);
    request.onerror = () =>
      reject(request.error ?? new Error(`Could not open ${dbName}`));
    request.onsuccess = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("urls")) {
        db.close();
        resolve();
        return;
      }
      const tx = db.transaction("urls", "readwrite");
      const store = tx.objectStore("urls");
      const deleteRequest = store.delete(key);
      deleteRequest.onsuccess = () => {
        db.close();
        resolve();
      };
      deleteRequest.onerror = () => {
        db.close();
        reject(deleteRequest.error ?? new Error(`Could not delete ${key}`));
      };
    };
  });
}

export function matchesModelCacheTarget(
  cacheKey: string,
  target: ModelCacheTarget,
): boolean {
  const decodedKey = safelyDecode(cacheKey);
  const matchers = new Set<string>([
    target.modelId,
    ...target.repoLabels,
    ...target.artifactUrls,
    ...target.artifactUrls.map((url) => safelyDecode(url)),
  ]);

  for (const rawMatcher of matchers) {
    const matcher = rawMatcher.trim();
    if (matcher && decodedKey.includes(matcher)) return true;
  }
  return false;
}

function safelyDecode(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}
