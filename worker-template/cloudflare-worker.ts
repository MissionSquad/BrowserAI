import { MODEL_PRESETS } from "@missionsquad/browserai";

/**
 * Cloudflare Worker template: same-origin model-artifact proxy for @missionsquad/browserai.
 *
 * Deploy this Worker (plus your static assets) and configure the SDK with
 * `{ modelSource: "proxy" }` (same origin) or `{ modelSource: "proxy", proxyOrigin: "https://…" }`.
 * Routes: /hf/* (WebLLM weights), /hf-transformers/* (Transformers.js artifacts),
 * /gh-raw/* (WebLLM wasm libraries), /__worker-health, /__proxy-test.
 *
 * Proxying is restricted to an allowlist: the MLC weight repos below plus every Transformers.js
 * preset in the SDK catalog (derived automatically, so it cannot drift out of sync).
 */

type AssetsBinding = {
  fetch(request: Request): Promise<Response>;
};

type Env = {
  ASSETS?: AssetsBinding;
};

const WORKER_VERSION = "0.1.0";
const WORKER_NAME = "browserai-proxy";

// WebLLM/MLC weight repos (fetched via /hf/*). The prebuilt records live under mlc-ai; the custom
// Gemma 4 MLC builds under welcoma. These are not derivable from MODEL_PRESETS (preset ids are
// WebLLM model ids, not HF repo paths), so they stay explicit.
const ALLOWED_HF_REPOS_BY_OWNER = new Map<string, Set<string>>([
  [
    "mlc-ai",
    new Set([
      "Qwen3.5-0.8B-q4f16_1-MLC",
      "Qwen3.5-2B-q4f16_1-MLC",
      "Qwen3.5-4B-q4f16_1-MLC",
      "gemma3-1b-it-q4f16_1-MLC",
      "Llama-3.2-1B-Instruct-q4f16_1-MLC",
      "Llama-3.2-3B-Instruct-q4f16_1-MLC",
      "Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
      "OLMo-2-0425-1B-Instruct-q4f16_1-MLC",
      "Phi-4-mini-instruct-q4f16_1-MLC",
      "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC",
      "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    ]),
  ],
  ["welcoma", new Set(["gemma-4-E2B-it-q4f16_1-MLC", "gemma-4-E4B-it-q4f16_1-MLC"])],
]);

// Every Transformers.js preset downloads from its own Hugging Face repo (preset id = owner/repo)
// through /hf-transformers/*. Derive those allowlist entries from the SDK's model catalog so the
// proxy can never drift out of sync with it — a missing entry 403s hosted model loads.
for (const preset of MODEL_PRESETS) {
  if (preset.backend !== "transformers-js") continue;
  const [owner, ...repoParts] = preset.id.split("/");
  if (!owner || repoParts.length === 0) continue;
  const repos = ALLOWED_HF_REPOS_BY_OWNER.get(owner) ?? new Set<string>();
  repos.add(repoParts.join("/"));
  ALLOWED_HF_REPOS_BY_OWNER.set(owner, repos);
}

const ALLOWED_GITHUB_OWNER = "mlc-ai";
const ALLOWED_GITHUB_REPO = "binary-mlc-llm-libs";
const ALLOWED_GITHUB_BRANCH = "main";
const ALLOWED_GITHUB_PREFIX = "web-llm-models/";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
  "Access-Control-Allow-Headers": "Range, If-None-Match, If-Modified-Since, Accept, Content-Type",
  "Access-Control-Expose-Headers":
    "Content-Length, Content-Range, Accept-Ranges, ETag, Last-Modified, X-Linked-Size, X-Linked-ETag, X-Repo-Commit, X-Upstream-URL, X-Upstream-Status, X-Proxy-Worker, X-Proxy-Route",
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/__worker-health") {
      return jsonResponse(
        {
          ok: true,
          worker: `${WORKER_NAME}/${WORKER_VERSION}`,
          app: WORKER_NAME,
          version: WORKER_VERSION,
          routes: ["/hf/*", "/hf-transformers/*", "/gh-raw/*", "/__proxy-test"],
          allowedHuggingFaceRepos: Array.from(ALLOWED_HF_REPOS_BY_OWNER.entries()).flatMap(([owner, repos]) =>
            Array.from(repos).map((repo) => `${owner}/${repo}`),
          ),
          note: "If this endpoint returns 200, the proxy Worker is deployed. If /hf/* returns 404 without X-Proxy-Worker, the request is bypassing the Worker.",
        },
        200,
        { "Cache-Control": "no-store" },
      );
    }

    if (url.pathname === "/__proxy-test") {
      return proxyTest(request, url);
    }

    if (url.pathname.startsWith("/hf/")) {
      return proxyHuggingFaceModelFile(request, url, "hf");
    }

    if (url.pathname.startsWith("/hf-transformers/")) {
      return proxyHuggingFaceModelFile(request, url, "hf-transformers");
    }

    if (url.pathname.startsWith("/gh-raw/")) {
      return proxyRawGithubModelLibrary(request, url);
    }

    return serveStaticAsset(request, env);
  },
};

async function serveStaticAsset(request: Request, env: Env): Promise<Response> {
  if (!env.ASSETS) {
    return jsonResponse({ error: "No static assets binding; this Worker only serves the proxy routes." }, 404);
  }
  const response = await env.ASSETS.fetch(request);
  const headers = new Headers(response.headers);
  const url = new URL(request.url);
  const accept = request.headers.get("Accept") ?? "";

  headers.set("X-App-Version", `${WORKER_NAME}/${WORKER_VERSION}`);

  // Hashed JS/CSS assets can be cached; keep the HTML shell fresh so users get new deploys.
  const looksLikeHtmlDocument =
    request.method === "GET" && (url.pathname === "/" || url.pathname.endsWith("/index.html") || accept.includes("text/html"));
  if (looksLikeHtmlDocument) {
    headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
    headers.set("Pragma", "no-cache");
    headers.set("Expires", "0");
  }

  return new Response(request.method === "HEAD" ? null : response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function isAllowedHuggingFaceRepo(owner: string, repo: string): boolean {
  return ALLOWED_HF_REPOS_BY_OWNER.get(owner)?.has(repo) === true;
}

async function proxyTest(request: Request, url: URL): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: withProxyHeaders(CORS_HEADERS, "proxy-test") });
  }

  if (request.method !== "GET" && request.method !== "HEAD") {
    return jsonResponse({ error: "Method not allowed" }, 405, {}, "proxy-test");
  }

  const owner = url.searchParams.get("owner") || "mlc-ai";
  const repo = url.searchParams.get("repo") || "";
  const file = url.searchParams.get("file") || "mlc-chat-config.json";

  if (!isAllowedHuggingFaceRepo(owner, repo)) {
    return jsonResponse({ ok: false, error: "Model repository is not allowlisted", owner, repo }, 403, {}, "proxy-test");
  }

  const filePath = file
    .split("/")
    .filter(Boolean)
    .map((part) => encodeURIComponent(decodeURIComponent(part)))
    .join("/");

  if (!filePath || filePath.includes("..")) {
    return jsonResponse({ ok: false, error: "Invalid file path", file }, 400, {}, "proxy-test");
  }

  const upstreamUrl = new URL(`https://huggingface.co/${owner}/${repo}/resolve/main/${filePath}`);
  const upstreamResponse = await fetch(upstreamUrl, { method: "HEAD", redirect: "follow" });

  return jsonResponse(
    {
      ok: upstreamResponse.ok,
      owner,
      repo,
      file: filePath,
      upstreamUrl: upstreamUrl.href,
      upstreamStatus: upstreamResponse.status,
      upstreamStatusText: upstreamResponse.statusText,
      contentType: upstreamResponse.headers.get("Content-Type"),
      contentLength: upstreamResponse.headers.get("Content-Length"),
    },
    upstreamResponse.ok ? 200 : 502,
    {
      "Cache-Control": "no-store",
      "X-Upstream-URL": upstreamUrl.href,
      "X-Upstream-Status": String(upstreamResponse.status),
    },
    "proxy-test",
  );
}

async function proxyHuggingFaceModelFile(request: Request, url: URL, route: "hf" | "hf-transformers"): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: withProxyHeaders(CORS_HEADERS, route) });
  }

  if (request.method !== "GET" && request.method !== "HEAD") {
    return jsonResponse({ error: "Method not allowed" }, 405, {}, route);
  }

  const match = url.pathname.match(/^\/(?:hf|hf-transformers)\/([^/]+)\/([^/]+)\/resolve\/([^/]+)\/(.+)$/);
  if (!match) {
    return jsonResponse(
      {
        error: "Invalid Hugging Face proxy path",
        expected: "/hf/:owner/:repo/resolve/:branch/:file",
      },
      400,
      {},
      route,
    );
  }

  const [, rawOwner, rawRepo, rawBranch, rawFilePath] = match;
  const owner = decodeURIComponent(rawOwner);
  const repo = decodeURIComponent(rawRepo);
  const branch = decodeURIComponent(rawBranch);
  const filePath = decodeAndReencodePath(rawFilePath);

  if (!isAllowedHuggingFaceRepo(owner, repo)) {
    return jsonResponse({ error: "Model repository is not allowlisted", owner, repo }, 403, {}, route);
  }

  const upstreamUrl = new URL(`https://huggingface.co/${owner}/${repo}/resolve/${encodeURIComponent(branch)}/${filePath}`);
  upstreamUrl.search = url.search;

  return proxyUpstream(request, upstreamUrl, route);
}

async function proxyRawGithubModelLibrary(request: Request, url: URL): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: withProxyHeaders(CORS_HEADERS, "gh-raw") });
  }

  if (request.method !== "GET" && request.method !== "HEAD") {
    return jsonResponse({ error: "Method not allowed" }, 405, {}, "gh-raw");
  }

  const match = url.pathname.match(/^\/gh-raw\/([^/]+)\/([^/]+)\/([^/]+)\/(.+)$/);
  if (!match) {
    return jsonResponse(
      {
        error: "Invalid raw GitHub proxy path",
        expected: "/gh-raw/:owner/:repo/:branch/:file",
      },
      400,
      {},
      "gh-raw",
    );
  }

  const [, rawOwner, rawRepo, rawBranch, rawFilePath] = match;
  const owner = decodeURIComponent(rawOwner);
  const repo = decodeURIComponent(rawRepo);
  const branch = decodeURIComponent(rawBranch);
  const filePath = rawFilePath
    .split("/")
    .map((part) => decodeURIComponent(part))
    .join("/");

  if (
    owner !== ALLOWED_GITHUB_OWNER ||
    repo !== ALLOWED_GITHUB_REPO ||
    branch !== ALLOWED_GITHUB_BRANCH ||
    !filePath.startsWith(ALLOWED_GITHUB_PREFIX) ||
    !filePath.endsWith(".wasm")
  ) {
    return jsonResponse({ error: "Raw GitHub file is not allowlisted", owner, repo, branch, filePath }, 403, {}, "gh-raw");
  }

  const encodedFilePath = filePath.split("/").map(encodeURIComponent).join("/");
  const upstreamUrl = new URL(`https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${encodedFilePath}`);
  upstreamUrl.search = url.search;

  return proxyUpstream(request, upstreamUrl, "gh-raw");
}

async function proxyUpstream(request: Request, upstreamUrl: URL, source: "hf" | "hf-transformers" | "gh-raw"): Promise<Response> {
  const upstreamHeaders = new Headers();
  copyHeader(request.headers, upstreamHeaders, "Accept");
  copyHeader(request.headers, upstreamHeaders, "Range");

  // Do not forward browser conditional request headers. If Hugging Face or GitHub returns 304,
  // WebLLM's artifact fetch/cache path can treat it as a failed artifact load. A fresh 200/206
  // response is clearer across Cache API and IndexedDB backends.
  const upstreamResponse = await fetch(upstreamUrl, {
    method: request.method,
    headers: upstreamHeaders,
    redirect: "follow",
  });

  if (!upstreamResponse.ok) {
    return jsonResponse(
      {
        error: "Upstream artifact request failed",
        source,
        upstreamUrl: upstreamUrl.href,
        upstreamStatus: upstreamResponse.status,
        upstreamStatusText: upstreamResponse.statusText,
      },
      upstreamResponse.status,
      {
        "X-Upstream-URL": upstreamUrl.href,
        "X-Upstream-Status": String(upstreamResponse.status),
        "Cache-Control": "no-store",
      },
      source,
    );
  }

  const responseHeaders = sanitizeProxyHeaders(upstreamResponse.headers);
  addCorsHeaders(responseHeaders);
  addProxyIdentityHeaders(responseHeaders, source);

  responseHeaders.set("Cache-Control", responseHeaders.get("Cache-Control") || "public, max-age=31536000, immutable");
  responseHeaders.set("Cross-Origin-Resource-Policy", "cross-origin");
  responseHeaders.set("X-Upstream-URL", upstreamUrl.href);
  responseHeaders.set("X-Upstream-Status", String(upstreamResponse.status));

  return new Response(request.method === "HEAD" ? null : upstreamResponse.body, {
    status: upstreamResponse.status,
    statusText: upstreamResponse.statusText,
    headers: responseHeaders,
  });
}

function sanitizeProxyHeaders(headers: Headers): Headers {
  const result = new Headers(headers);

  for (const name of [
    "Set-Cookie",
    "Vary",
    "Report-To",
    "NEL",
    "Content-Security-Policy",
    "Content-Security-Policy-Report-Only",
    "X-Frame-Options",
  ]) {
    result.delete(name);
  }

  return result;
}

function decodeAndReencodePath(rawPath: string): string {
  return rawPath
    .split("/")
    .map((part) => decodeURIComponent(part))
    .map(encodeURIComponent)
    .join("/");
}

function copyHeader(source: Headers, target: Headers, name: string): void {
  const value = source.get(name);
  if (value) {
    target.set(name, value);
  }
}

function addCorsHeaders(headers: Headers): void {
  for (const [key, value] of Object.entries(CORS_HEADERS)) {
    headers.set(key, value);
  }
}

function addProxyIdentityHeaders(headers: Headers, route = "worker"): void {
  headers.set("X-Proxy-Worker", `${WORKER_NAME}/${WORKER_VERSION}`);
  headers.set("X-Proxy-Route", route);
}

function withProxyHeaders(headersLike: Record<string, string>, route = "worker"): Headers {
  const headers = new Headers(headersLike);
  addProxyIdentityHeaders(headers, route);
  return headers;
}

function jsonResponse(body: unknown, status: number, extraHeaders: Record<string, string> = {}, route = "worker"): Response {
  const headers = new Headers({ "Content-Type": "application/json; charset=utf-8", ...extraHeaders });
  addCorsHeaders(headers);
  addProxyIdentityHeaders(headers, route);
  return new Response(JSON.stringify(body, null, 2), { status, headers });
}
