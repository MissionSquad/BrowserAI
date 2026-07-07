# BrowserAI proxy Worker template

By default the SDK downloads model artifacts **directly** from Hugging Face (and raw GitHub for
WebLLM wasm libraries), which works on any site. Hosted deployments sometimes hit CORS or
cache-storage edge cases with those cross-origin downloads; this Cloudflare Worker makes every
artifact URL same-origin and restricts proxying to the SDK's model catalog.

## Deploy

1. Copy `cloudflare-worker.ts` and `wrangler.jsonc` into your project (the Worker imports
   `MODEL_PRESETS` from `@missionsquad/browserai`, so the package must be installed).
2. Adjust `wrangler.jsonc` (`name`, optionally `routes` for a custom domain, and
   `assets.directory` if the Worker should also serve your site).
3. `npx wrangler deploy`
4. Verify: `https://<your-worker-domain>/__worker-health` should return
   `{ ok: true, worker: "browserai-proxy/…" }` and list every Transformers.js preset under
   `allowedHuggingFaceRepos`.

## Point the SDK at it

```ts
import { BrowserAI } from "@missionsquad/browserai";

// Same origin (the Worker serves your site too):
const ai = new BrowserAI({ modelSource: "proxy" });

// Separate proxy origin:
const ai2 = new BrowserAI({ modelSource: "proxy", proxyOrigin: "https://browserai-proxy.example.workers.dev" });
```

The SDK probes `/__worker-health` plus the selected model's config before each proxied load
(`verifyProxy: false` disables the probe). Kokoro is the one exception to proxying: kokoro-js
ships its own runtime and fetches its weights directly from its CDN/Hugging Face.

## Allowlist

`/hf-transformers/*` entries are derived from the SDK catalog at build time, so they never drift.
The `/hf/*` (MLC weights) and `/gh-raw/*` (wasm libraries) allowlists are explicit constants at the
top of `cloudflare-worker.ts` — extend them if you add custom MLC records.
