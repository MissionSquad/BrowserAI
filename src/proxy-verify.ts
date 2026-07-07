import type { AppConfig, ModelRecord } from "@mlc-ai/web-llm";
import { ProxyVerificationError } from "./errors.js";

/**
 * Pre-load health checks for a deployed proxy Worker (the `worker-template/` in this package).
 * Failing fast with an actionable message beats letting an engine surface an opaque artifact
 * fetch error hundreds of megabytes into a download.
 */

/** The X-Proxy-Worker prefixes accepted as "this looks like a compatible proxy Worker". */
const KNOWN_WORKER_PREFIXES = ["browserai-proxy/", "browser-json-llm-app/"];

async function fetchWorkerHealth(proxyOrigin: string): Promise<Response> {
  const healthResponse = await fetch(new URL("/__worker-health", proxyOrigin), { cache: "no-store" });
  if (!healthResponse.ok) {
    throw new ProxyVerificationError(
      `Proxy Worker is not active: ${proxyOrigin}/__worker-health returned ${healthResponse.status}. Deploy the worker-template with \`npx wrangler deploy\`.`,
    );
  }
  return healthResponse;
}

/** Verify the proxy Worker is live and can serve the selected WebLLM model's config + wasm library. */
export async function verifyWebLLMProxy(proxyOrigin: string, modelId: string, appConfig: AppConfig): Promise<void> {
  const healthResponse = await fetchWorkerHealth(proxyOrigin);
  const healthText = await healthResponse.text().catch(() => "");
  let healthBody: unknown = healthText;
  try {
    healthBody = JSON.parse(healthText);
  } catch {
    // Static HTML/old Worker response stays raw for diagnostics.
  }
  const workerHeader = healthResponse.headers.get("X-Proxy-Worker") ?? "";
  const healthObject = typeof healthBody === "object" && healthBody !== null ? (healthBody as Record<string, unknown>) : {};
  const workerLooksCurrent =
    KNOWN_WORKER_PREFIXES.some((prefix) => workerHeader.startsWith(prefix)) ||
    (healthObject.ok === true && typeof healthObject.version === "string");
  if (!workerLooksCurrent) {
    throw new ProxyVerificationError(
      `Proxy Worker health check did not look like a compatible Worker. Status ${healthResponse.status}, X-Proxy-Worker: ${workerHeader || "missing"}.`,
    );
  }

  const selectedRecord = appConfig.model_list.find((record) => record.model_id === modelId);
  if (!selectedRecord) {
    throw new ProxyVerificationError(`Model ${modelId} is not present in the WebLLM AppConfig.`);
  }

  const configProbeUrl = buildModelConfigProbeUrl(selectedRecord, proxyOrigin);
  configProbeUrl.searchParams.set("probe", String(Date.now()));
  const configProbeResponse = await fetch(configProbeUrl, { cache: "no-store" });
  const configProxyHeader = configProbeResponse.headers.get("X-Proxy-Worker");
  if (!configProbeResponse.ok || (configProbeUrl.origin === new URL(proxyOrigin).origin && !configProxyHeader)) {
    throw new ProxyVerificationError(
      `Proxy /hf model-config probe failed for ${modelId}: ${configProbeResponse.status}. URL: ${configProbeUrl.pathname}. ${await readResponseBodyForError(configProbeResponse)}`,
    );
  }

  const modelLib = selectedRecord.model_lib;
  if (modelLib && modelLib.startsWith(proxyOrigin)) {
    const modelLibProbeUrl = new URL(modelLib);
    modelLibProbeUrl.searchParams.set("probe", String(Date.now()));
    const modelLibProbeResponse = await fetch(modelLibProbeUrl, { method: "HEAD", cache: "no-store" });
    const libProxyHeader = modelLibProbeResponse.headers.get("X-Proxy-Worker");
    if (!modelLibProbeResponse.ok || !libProxyHeader) {
      throw new ProxyVerificationError(
        `Proxy model-library probe failed for ${modelId}: ${modelLibProbeResponse.status}. URL: ${modelLibProbeUrl.pathname}.`,
      );
    }
  }
}

function buildModelConfigProbeUrl(record: ModelRecord, proxyOrigin: string): URL {
  const modelUrl = new URL(record.model, proxyOrigin);
  const path = modelUrl.pathname.replace(/\/+$/, "");
  if (/\/resolve\/[^/]+(?:\/|$)/.test(path)) {
    const base = new URL(modelUrl.href.endsWith("/") ? modelUrl.href : `${modelUrl.href}/`);
    return new URL("mlc-chat-config.json", base);
  }
  return new URL(`${path}/resolve/main/mlc-chat-config.json`, modelUrl.origin);
}

/** Verify the proxy Worker can serve a Transformers.js repo's config through /hf-transformers/. */
export async function verifyTransformersProxy(proxyOrigin: string, modelId: string): Promise<void> {
  await fetchWorkerHealth(proxyOrigin);
  const { owner, repo } = splitHuggingFaceRepoId(modelId);
  let lastStatus = "not fetched";
  for (const file of ["generation_config.json", "config.json"]) {
    const url = new URL(`/hf-transformers/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/resolve/main/${file}`, proxyOrigin);
    url.searchParams.set("probe", String(Date.now()));
    const response = await fetch(url, { method: "HEAD", cache: "no-store" });
    const proxyHeader = response.headers.get("X-Proxy-Worker");
    lastStatus = `${url.pathname}: ${response.status}, X-Proxy-Worker=${proxyHeader ?? "missing"}`;
    if (response.ok && proxyHeader) return;
  }
  throw new ProxyVerificationError(`Proxy Transformers.js probe failed for ${modelId}. Last status: ${lastStatus}`);
}

export function splitHuggingFaceRepoId(modelId: string): { owner: string; repo: string } {
  const parts = modelId.split("/");
  if (parts.length < 2) throw new ProxyVerificationError(`Expected a Hugging Face repo ID in owner/repo form, got ${modelId}`);
  return { owner: parts[0], repo: parts.slice(1).join("/") };
}

async function readResponseBodyForError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return `${response.status} ${response.statusText}`;
  try {
    return JSON.stringify(JSON.parse(text));
  } catch {
    return text.slice(0, 300);
  }
}
