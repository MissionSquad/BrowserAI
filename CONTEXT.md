# CONTEXT.md — architecture and codebase guide for `@missionsquad/browserai`

This file gives an AI coding assistant (any provider) or a new engineer the context needed to work
on this repository: what the package is, how it is architected, what each file does, the invariants
that must not be broken, and how to build, test, and release it. Facts here are derived from the
source; when this file and the code disagree, the code wins — update this file in the same change.

## What this package is

`@missionsquad/browserai` is an **embeddable, UI-free TypeScript SDK** that runs small open AI
models **entirely in the browser** on WebGPU — no inference server. It provides a curated 47-model
catalog, model downloading with progress + browser caching, a capability-slot manager (at most one
model per slot: `text`, `vision`, `stt`, `tts`), three inference runtimes, streaming, schema-
constrained JSON generation, run metrics/history, cache management, and a headless press-to-talk
voice-assistant pipeline.

Hard facts:

- ESM-only (`"type": "module"`), built with `tsc` to `dist/` (ESM + `.d.ts` + source maps).
- Runtime environment is the **browser** (WebGPU, Web Audio, Cache API, IndexedDB). Node.js ≥ 20 is
  required only for development; tests run in Node against pure logic.
- Two package entry points: `"."` (the full SDK, `dist/index.js`) and `"./worker"`
  (`dist/webllm.worker.js`, a WebLLM web-worker entry a host app re-exports from its own worker
  file).
- The three inference runtimes:
  1. **WebLLM (MLC)** — quantized MLC text models via `@mlc-ai/web-llm`. Supports real
     grammar-constrained JSON decoding and detailed engine telemetry. Runs on the main thread by
     default, or in a host-bundled worker.
  2. **Transformers.js (ONNX)** — via `@huggingface/transformers`: ONNX text LLMs, ASR
     (Whisper/Moonshine/Parakeet/…), Supertonic speech, vision-language models, Gemma 4
     multimodal (text+image+audio), and audio-LLM/realtime ASR.
  3. **kokoro-js** — Kokoro 82M TTS, loaded at runtime from a CDN ESM bundle by default
     (configurable URL or host-bundled loader). Supports sentence-by-sentence streaming synthesis.
- `@mlc-ai/web-llm` (0.2.84) and `@huggingface/transformers` (4.2.0) are regular dependencies, but
  the main entry loads them **only via dynamic `import()` at model-load time**, so a host's
  initial bundle stays small. The single static import of `@mlc-ai/web-llm` in shipped code is
  `src/webllm.worker.ts` — the `./worker` entry, which a host bundles into its own worker file.
  `kokoro-js` is deliberately **not** a dependency (runtime CDN import).

## Big-picture architecture

```
                       ┌────────────────────────────────────────────────┐
                       │                 BrowserAI (client.ts)          │
 host app ── config ──▶│  catalog · slots · load/unload · run serializer│
                       │  generateText / transcribe / synthesize /     │
                       │  describeImage · cache mgmt · history         │
                       └───────┬──────────────┬──────────────┬─────────┘
                               │              │              │
                    WebLLM engine     Transformers.js    kokoro-js (CDN)
                 (main thread or      pipelines/models   streaming TTS
                  host worker via     (inference.ts +
                  ./worker entry)      multimodal.ts)
                               │              │              │
                       ┌───────┴──────────────┴──────────────┴─────────┐
                       │  model artifacts: Hugging Face / raw GitHub    │
                       │  direct (default) or via the Cloudflare proxy  │
                       │  Worker (worker-template/), cached in browser  │
                       │  storage (IndexedDB / Cache API / OPFS)        │
                       └────────────────────────────────────────────────┘

 VoicePipeline (voice/voice-pipeline.ts) sits ON TOP of BrowserAI:
 MicRecorder → transcribe → generateText → streaming TTS or clip → speakers,
 as a typed-event state machine (idle/listening/transcribing/thinking/speaking/error).
```

Layering rules:

- `client.ts` owns all **live engine references** and passes them into **stateless** helpers:
  `inference.ts` (`generate(handles, …)`) and `multimodal.ts` (functions over a `MultimodalHandle`
  discriminated union). Those modules never hold state.
- `VoicePipeline` never talks to engines directly — every model call goes through the `BrowserAI`
  client, so slot lookup, serialization, and error taxonomy stay in one place.
- Pure logic (catalog, JSON parsing, schema validation, PCM math, history, URL rewriting, voice
  helpers) is kept dependency-free and DOM-free where possible so it is unit-testable in Node.

## Repository layout

```
src/
  index.ts              Barrel entry: the complete public API surface (re-exports only).
  client.ts             BrowserAI class — slots, loaders, run serialization, facade (785 lines).
  config.ts             BrowserAIConfig / ResolvedConfig types + resolveConfig defaults.
  errors.ts             BrowserAIError hierarchy (7 subclasses) + formatError.
  events.ts             TypedEmitter<Events> — minimal typed event emitter (protected emit).
  models.ts             47-preset catalog, slot helpers, TTS voice rosters, MLC record overrides.
  generation.ts         RuntimeParameters (jsonMode, latencyBreakdown) + buildResponseFormat.
  inference.ts          Stateless text generation for WebLLM + Transformers.js; RuntimeStats.
  multimodal.ts         STT / TTS / VLM / Gemma / audio-LLM recipes over Transformers.js + kokoro.
  prompt.ts             buildExtractionMessages — system+user prompt embedding a JSON Schema.
  schema.ts             DEFAULT_EXTRACTION_SCHEMA + stringifySchema + stripSchemaMetadata.
  schema-validator.ts   Dependency-free JSON Schema subset validator (the SDK's only validator).
  json.ts               parseJsonFromModel — fence/think stripping, balanced-JSON recovery.
  hardware.ts           collectHardwareSnapshot (WebGPU probe) + formatting + VRAM hints.
  history.ts            RunHistoryStore (injected StorageLike) + per-model aggregation.
  metrics.ts            GenStats→RuntimeStats and RuntimeStats→RunRecord adapters.
  model-cache.ts        Inspect/delete model artifacts across Cache API / IndexedDB / OPFS.
  model-source.ts       WebLLM AppConfig assembly; direct vs proxy URL rewriting (/hf/, /gh-raw/).
  proxy-verify.ts       Pre-load health probes against a deployed proxy Worker.
  text-utils.ts         stripThinkTags (streaming-safe <think> removal).
  webllm.worker.ts      9-line worker entry: WebWorkerMLCEngineHandler + self.onmessage.
  audio/
    pcm.ts              AUDIO_SR = 16000, pitchShiftBuffer (granular OLA), encodeWavBlob.
    decode.ts           decodeAudioTo16kMono (AudioContext decode → OfflineAudioContext resample).
    recorder.ts         MicRecorder — press-to-talk getUserMedia/MediaRecorder → 16 kHz PCM.
    streaming-player.ts createStreamingAudioPlayer — gapless chunk scheduler for streamed TTS.
    clip.ts             SynthClip packaging — blob/object-URL clips, pitch baking, playback rate.
  voice/
    helpers.ts          Pure pipeline helpers: recommended models, history cap, streamedDelta.
    voice-pipeline.ts   VoicePipeline — the press-to-talk state machine (14 typed events).
test/                   9 vitest files, 89 tests (see "Testing" below).
examples/               Vite-served runnable demos (5 pages) aliasing the SDK to ../src.
worker-template/        Copyable Cloudflare Worker: model-artifact proxy + wrangler.jsonc + README.
.github/workflows/      ci.yml (PR build+test) and publish.yml (version-guarded npm publish).
tasks/todo.md           Historical build plan/review notes from the original extraction.
```

## Core concepts and invariants

### Capability slots

Every model preset maps to one or more of the four `ModelTask` slots — `text`, `vision`, `stt`,
`tts` (`modelSlots(preset)`, ordered by `SLOT_ORDER`). **At most one loaded model per slot**, so up
to four models can be resident. Loading a model whose slots conflict with a loaded model evicts the
conflicting owner (whole-model eviction — a multi-slot owner is fully released even for a one-slot
conflict). The invariant is enforced only inside `load()`; `slotOwner()`/`occupiedSlots()` assume
it. Note: image-text VLM presets deliberately do **not** list `text` in their tasks (their
Transformers.js processors reject text-only input), so a VLM occupies only the `vision` slot; only
real text runtimes and the Gemma-4 multimodal ONNX presets fill the `text` slot.

### Two-level concurrency control (client.ts)

1. **Lifecycle mutex** — `#lifecycleBusy` is claimed *synchronously* (before any `await`) by
   `load`, `unload`, `unloadAll`, `deleteModelArtifacts`, `deleteAllModelArtifacts`. Concurrent
   lifecycle calls **throw** (`"Another load/unload is already in progress."`) rather than queue.
2. **Per-model run chains** — `#runChains` is a `WeakMap<LoadedModel, Promise>`; every inference
   call *and* model teardown chains onto the model's promise via `#runExclusive`. Rejections are
   swallowed in the stored chain (a failed run never poisons the next), while callers still see the
   rejection. Teardown going through the same chain means unload/evict waits out in-flight runs.
   `createStreamingTts` parks a gate promise in the chain for the whole streaming session and
   releases it exactly at synthesis-drain — error paths must close the splitter/release the session
   or the model's chain deadlocks.

### Model loading, sources, and caching

- Backend selection: `preset.mmRuntime` set → multimodal loader; else `preset.backend ===
  "transformers-js"` → Transformers.js text pipeline; else WebLLM engine.
- WebGPU is probed on every load (`collectHardwareSnapshot`). Without WebGPU, only
  `mmRuntime ∈ { "stt", "tts-pipeline", "tts-kokoro" }` falls back to `wasm`; everything else
  throws `WebGPUUnavailableError`.
- **Model source** is `"direct"` (default; straight from `huggingface.co` and
  `raw.githubusercontent.com`) or `"proxy"` (URL-rewritten to a deployed copy of
  `worker-template/`): `huggingface.co` URLs → `/hf/{owner}/{repo}/…` and
  `raw.githubusercontent.com` URLs → `/gh-raw/{owner}/{repo}/{branch}/…` — model records are
  rewritten by host, so WebLLM's bundled wasm libs (raw GitHub) become `/gh-raw/…` while the two
  custom Gemma-4 builds' HF-hosted wasm libs become `/hf/…`. Transformers.js artifacts →
  `env.remoteHost = "{origin}/hf-transformers/"` (that rewrite lives in client.ts, not
  model-source.ts). In proxy
  mode, `proxy-verify.ts` probes `/__worker-health` + the model's config before each load
  (disable with `verifyProxy: false`); Kokoro is exempt (kokoro-js fetches its own weights).
- **Caching**: WebLLM artifacts go to the backend selected by `cacheBackend` (default
  `"indexeddb"`) — the storage names are `webllm/model`, `webllm/config`, `webllm/wasm` (used as
  both Cache API bucket names and IndexedDB database names) and OPFS root `tvmjs-opfs-store`.
  Transformers.js artifacts always go to the Cache API bucket `transformers-cache` regardless of
  `cacheBackend`. `model-cache.ts` inspects/deletes exactly these names and never enumerates or
  touches unrelated storage (no `caches.keys()` sweeps, no localStorage/cookies). The literal
  `"transformers-cache"` is duplicated as `TRANSFORMERS_CACHE_KEY` (model-source.ts) and
  `TRANSFORMERS_CACHE_SCOPE` (model-cache.ts) — keep them in sync.
- Context-length override (WebLLM only): `normalizeContextLength` maps null/invalid/default-4096 to
  "no override", otherwise clamps to [512, 32768] and merges `context_window_size` into the model
  record's overrides (preserving curated overrides like Gemma's fixed-context settings).

### The model catalog (models.ts)

`MODEL_PRESETS` holds 47 presets: 11 `webllm-prebuilt`, 2 `custom-mlc` (Gemma 4 E2B/E4B MLC builds
hosted on `welcoma/…`, appended to WebLLM's AppConfig because 0.2.84 doesn't include them), and 34
`transformers-onnx`. By capability: 26 text LLMs, 10 STT, 3 TTS (Kokoro, Supertonic 1/2), 6 vision,
2 Gemma-4 multimodal (tasks `["text","vision","stt"]`). Each preset carries GPU-memory estimates
(`vramRequiredMB`, `vramSource: "webllm" | "estimated"`), download-size text, stability
(`recommended | stable | experimental`), a `defaultRuntime` (RuntimePreset) derived from a
deterministic extraction base (`maxTokens 512, temperature 0, topP 0.85, repetitionPenalty 1.08,
frequencyPenalty 0.2, presencePenalty 0, seed 42`) with per-model tweaks, and optional
`transformersDtype` / `mmRuntime` / modality metadata. TTS voice rosters are disjoint:
`KOKORO_VOICES` (6, e.g. `af_heart`) vs `SUPERTONIC_VOICES` (`F1`–`F5`, `M1`–`M5`);
`ttsVoicesForModel(modelId)` returns the right roster or `[]`. Gotcha: `getModelPreset(id)` falls
back to `MODEL_PRESETS[0]` for unknown ids (the client's `load()` throws `UnknownModelError` first;
`getPreset()` returns `undefined`).

### Text generation and JSON mode

`generateText` requires a `text`-slot owner. Gemma multimodal handles route to `generateGemmaText`
(Gemma-4 has no WebLLM engine / LM pipeline); everything else goes through the shared
`generate(handles, messages, runtime, preset, options)`:

- **WebLLM path**: `resetChat(false)` before each call unless `options.resetChat === false`
  (stateless — history is always passed explicitly). Streaming when `onDelta` is provided
  (`stream_options: { include_usage: true }`), blocking otherwise. `response_format` is built
  **only when `options.schema !== undefined`** via `buildResponseFormat(runtime.jsonMode, schema)`:
  `"none"` → undefined, `"json_object"` → `{ type: "json_object" }`, `"schema"` (default) →
  `{ type: "json_object", schema: JSON.stringify(stripSchemaMetadata(schema)) }` — genuine
  grammar-constrained decoding.
- **Transformers.js path**: prompt-guided only (the schema goes into the prompt via
  `buildExtractionMessages`, which embeds the *un-stripped* schema; only the constrained-decoder
  path strips metadata). A `TextStreamer` is attached even without `onDelta` because it is the
  token-telemetry source.
- `onDelta` **always receives the full accumulated text**, never an incremental delta — everywhere
  in the SDK (inference, multimodal, voice pipeline `replydelta`).
- Post-processing helpers: `parseJsonFromModel` (unwraps the first fenced block, strips completed
  `<think>` pairs, recovers the first balanced JSON value from noisy output, warns on
  prefix/suffix), `validateAgainstJsonSchema` (supported subset: `type` incl. type arrays +
  `integer`/`null`, `enum` (JSON.stringify equality), `min/maxLength`, `minimum/maximum`,
  `min/maxItems`, single-schema `items`, `properties`/`required`, `additionalProperties: false`;
  no `$ref`, `pattern`, `oneOf`, etc. — exactly what `DEFAULT_EXTRACTION_SCHEMA` needs),
  `stripThinkTags` (streaming-safe: removes closed pairs, truncates at an unclosed `<think>`).

### Telemetry and history

Every generation returns `RuntimeStats` (tokens, finish reason, measured elapsed/decode rate;
WebLLM adds engine-reported `extra` — decode tok/s, TTFT, per-token latency breakdown — and the
Transformers path *computes* equivalents from streamer timing so headline metrics never blank out).
Multimodal runs produce `GenStats` → `buildMultimodalStats` → `RuntimeStats`. `runRecordFromStats`
maps stats to a persistable `RunRecord` (returns `null` for error-placeholder stats — failed runs
never enter history) and attributes the run to the model that produced the stats.
`RunHistoryStore` persists newest-first with a cap (defaults: key `browserai:v1:history`, 200
records) into an injected `StorageLike` (defaults to `localStorage`, pass `storage: null` for
memory-only); all storage errors are swallowed so quota/private-mode failures degrade gracefully.
For ASR runs (which emit no token telemetry), the SDK exports `wordCountGenStats` as an explicit
opt-in helper — a host that wants such runs in history calls it itself to substitute word counts
for token counts; the SDK never performs that substitution automatically.

### Errors and events

All SDK errors are `BrowserAIError` subclasses (one level deep): `UnknownModelError`,
`WebGPUUnavailableError` (`.reason`), `ModelLoadError` (`.modelId`, `cause`),
`ProxyVerificationError`, `MissingModelError` (`.slot`), `MicrophoneError` (`.permissionDenied`
derived from DOMException name), `VoiceTurnError` (`.label` names the failing component because
the streaming voice path runs generation and synthesis concurrently — the stage alone can't
identify the failure). `load()` wraps any non-BrowserAIError into `ModelLoadError`. Branch with
`instanceof`.

Events use `TypedEmitter` (events.ts): `emit` is `protected` (only the SDK fires), listeners are
exception-isolated and iterated over a copy, `on()`/`once()` return unsubscribe functions.
`BrowserAIEvents`: `loadprogress` (0–1; Transformers' 0–100 is normalized and status-only events
re-emit the last numeric value so bars never snap back), `status`, `modelloaded`, `modelunloaded`.

### Audio utilities

- `AUDIO_SR = 16000` is the SDK-wide PCM sample rate; every STT/audio-LLM consumer expects 16 kHz
  mono `Float32Array`. Only the plain ASR-pipeline (`kind: "stt"`) accepts a URL input to
  `transcribe`; all other STT kinds require decoded PCM (use `decodeAudioTo16kMono`).
- `decodeAudioTo16kMono`, `capSrc` (canvas image downscale), `MicRecorder`, and clip playback use
  main-thread browser APIs (AudioContext, Image, canvas, MediaRecorder) — they do not work in
  workers or Node.
- `MicRecorder` hardening: `#pending` claimed synchronously so a double-press can't start two
  recorders; `cancel()` during a pending `getUserMedia` sets an abort flag so the mic is released
  when the prompt resolves; `stop()` handles auto-stopped recorders (device unplugged) and
  zero-byte blobs; host visualization hooks are try/catch-wrapped.
- Streaming player: one AudioContext + GainNode, butt-joined chunk scheduling (gaps, never
  overlap, if synthesis falls behind); `stop()` always resolves a parked `end()` so barge-in can
  never hang a turn. Clips: pitch is baked into the WAV blob offline via `pitchShiftBuffer`
  (original PCM kept on the clip for re-pitching); speed is applied **exactly once per engine** —
  Kokoro bakes it at synthesis (playbackRate 1), Supertonic gets it at playback
  (`clipPlaybackRate`). A media element allows exactly **one** `MediaElementAudioSourceNode` for
  its whole life — hosts must create an analyzer once and never close its context while the
  element is in use.

### VoicePipeline (voice/voice-pipeline.ts)

Headless press-to-talk state machine over a `BrowserAI` client. Stages: `idle → listening →
transcribing → thinking → speaking → idle`, with `error` reachable from any stage. 14 typed
events: `stagechange`, `recordingchange`, `micstream`/`micstreamended` (attach/dispose input
analyzers), `partialtranscript` (generative STT models only), `transcript`, `replydelta` (full
accumulated, think-stripped), `speechstart {mode: "streaming"|"clip"}`, `replyclip`,
`turncommitted`, `turndiscarded`, `error`, `playbackfinished`, `reset`.

Key behaviors:

- **Discard rules** (`turndiscarded`, history untouched): `contention` (client or external work
  busy after the mic stops), `too-short` (< `minTakeSamples`, default 4000 samples ≈ ¼ s),
  `no-speech` (empty transcript), `empty-reply` (fails `shouldSpeakReply` — blank or the
  `"[empty response]"` sentinel).
- **Streaming vs clip**: chosen by `client.supportsStreamingTts()` (Kokoro loaded → streaming;
  reply is spoken sentence-by-sentence *while the LLM is still generating*, only the monotonic new
  suffix (`streamedDelta`) is pushed to the splitter). Otherwise the finished reply is synthesized
  as one clip. Both paths go idle at synthesis-drain / playback-start, so the audible tail plays
  during `idle` and barge-in stays possible; `playbackfinished` fires after `turncommitted`.
- **Barge-in**: pressing talk stops playback first; the streaming consumer halts via a player
  identity check, and `playbackfinished` is suppressed for the interrupted player.
- **Commit-on-success only**: history is appended and `turncommitted` emitted only after a
  speakable reply; every failure path goes to `error` with a `VoiceTurnError` label and never
  touches history. The LLM sees system prompt + `capPipelineHistory` (default 6 turns) + the new
  user turn; turn runtime = preset defaults overlaid with `jsonMode: "none"`, `temperature 0.7`,
  `maxTokens 256`, `seed null`, then host overrides.
- `loadRecommendedModels()` sequentially loads whichever of `PIPELINE_RECOMMENDED` slots are empty:
  Whisper Base (`onnx-community/whisper-base`), Qwen3.5 0.8B (`Qwen3.5-0.8B-q4f16_1-MLC`), Kokoro
  82M (`onnx-community/Kokoro-82M-v1.0-ONNX`) — disjoint slots, ~2.5 GB VRAM total.
- `reset()` refuses while a turn is active; `dispose()` cancels the recorder, stops playback,
  removes the clip element's `ended` listener (host-supplied elements don't accumulate listeners
  across pipeline instances), and clears all listeners.

## Build system and TypeScript configuration

- **`moduleResolution: "Bundler"` — do not change to NodeNext.** `@mlc-ai/web-llm` is an ESM
  package with extensionless internal re-exports; under NodeNext its types silently degrade to
  `any`. Relative imports in `src/` still carry `.js` extensions so the emitted ESM works under
  both bundlers and native ESM. (tsconfig.json documents this; tasks/todo.md records the
  discovery.)
- Root `tsconfig.json`: type-checks everything (src, test, worker-template, vitest.config.ts),
  `strict`, `noEmit`, ES2022 target, DOM + WebWorker libs, and a `paths` alias mapping
  `@missionsquad/browserai` → `./src/index.ts`. `tsconfig.build.json` extends it: emits `src/`
  only to `dist/` with `declaration`, `sourceMap`, `declarationMap`, and **resets `paths` to `{}`**
  so the self-alias is never baked into the published output.
- The **alias-to-source convention** appears in four places and must stay consistent: root
  tsconfig `paths`, `vitest.config.ts` `resolve.alias`, `examples/vite.config.ts` `resolve.alias`,
  `examples/tsconfig.json` `paths`. It lets `worker-template/cloudflare-worker.ts` and the examples
  import the real published package name while resolving to `../src` in-repo (no build needed).
- `package.json` specifics:
  - `files`: `["dist", "src", "worker-template", "README.md"]` — `src` ships because emitted
    source maps reference `../src`.
  - `sideEffects: ["./dist/webllm.worker.js"]` — everything else is tree-shakable; the worker
    entry has **zero exports** and exists purely for module-level side effects, consumed as
    `export * from "@missionsquad/browserai/worker"`. Without this exemption bundlers would
    tree-shake the worker to an empty module and worker-mode loads would hang. Do not set
    `sideEffects: false`.
  - Scripts: `clean` (node `fs.rmSync` on dist) → `build` (clean + whole-repo `tsc --noEmit` +
    emit via `tsc -p tsconfig.build.json`) → `test` (`vitest run`) → `prepublishOnly`
    (build + test, the final publish gate).
- Adding a **static** top-level import of `@mlc-ai/web-llm` or `@huggingface/transformers`
  anywhere except `src/webllm.worker.ts` (or a type-only import) defeats the dynamic-import
  bundle-size strategy — keep engine imports inside the loaders in `client.ts`.

## Testing

`npm test` runs vitest in the Node environment against source (no build needed): 9 files, 89 tests.

| File | Covers | Tests |
| --- | --- | --- |
| test/client.test.ts | Slot manager (load/evict/unload/WebGPU gating) + inference facade (serialization, TTS lock) with mocked engines | 13 |
| test/audio.test.ts | Gemma audio windowing, voice rosters, Supertonic URLs, pitch shift, WAV encode | 15 |
| test/models.test.ts | Catalog invariants (unique ids, owner/repo-shaped ONNX ids that the proxy allowlist derives from, custom Gemma records) | 12 |
| test/json.test.ts | parseJsonFromModel, schema validation, stripThinkTags streaming behavior | 11 |
| test/voice-helpers.test.ts | missingPipelineModels, shouldSpeakReply, capPipelineHistory, streamedDelta | 10 |
| test/history.test.ts | RunHistoryStore persistence/aggregation, runRecordFromStats, wordCountGenStats | 10 |
| test/model-source.test.ts | /hf/ + /gh-raw/ rewriting, AppConfig merging, context override | 9 |
| test/worker-template.test.ts | Worker routes in-process: health, proxying, allowlist 403s | 5 |
| test/events.test.ts | TypedEmitter semantics | 4 |

## CI/CD

- `.github/workflows/ci.yml` — every PR (opened/reopened/synchronize; markdown-only changes are
  skipped via `paths-ignore`): `npm ci` → `npm run build` → `npm test` →
  `npx tsc -p examples/tsconfig.json`, on a Node 20 + 22 matrix (20 = `engines` floor, 22 =
  publish version).
- `.github/workflows/publish.yml` — every push to `main` (i.e. every PR merge; markdown-only
  merges skipped): build + test, then `npm view <name>@<version>` — if that version is not on the
  registry, `npm publish --access public` (scoped package) authenticated by the `NPM_TOKEN`
  secret. Merges without a version bump build + test and stop. A non-cancelling `npm-publish`
  concurrency group serializes publishes (GitHub keeps at most one pending run per group, so two
  rapid merges that each bump the version can skip the intermediate version — the newest always
  publishes).

## Examples (examples/)

Five self-contained, heavily commented demo pages served by Vite (`npm install && npm run dev` in
`examples/`), aliasing the SDK to `../src/index.ts` — no parent build needed:
`text-generation` (load + streaming + stats), `json-extraction` (schema-constrained decoding +
parse + validate), `speech` (MicRecorder → transcribe; synthesizeToClip → `<audio>`),
`voice-assistant` (VoicePipeline events end-to-end), `vision` (describeImage). WebGPU-capable
browser required for text/vision; STT/TTS pages can fall back to wasm. `optimizeDeps.exclude`
keeps the two heavy deps out of Vite pre-bundling (they are dynamic-imported at model load).

## Proxy worker template (worker-template/)

A copyable Cloudflare Worker for `{ modelSource: "proxy" }` deployments. Routes:
`/__worker-health` (JSON diagnostics incl. the full allowlist), `/__proxy-test`,
`/hf/:owner/:repo/resolve/:branch/:file` and `/hf-transformers/…` (same shape) proxied to
`huggingface.co`, `/gh-raw/:owner/:repo/:branch/:file` proxied to `raw.githubusercontent.com`
(hard-restricted to `mlc-ai/binary-mlc-llm-libs`, branch `main`, `web-llm-models/…*.wasm`), and
static assets via the `ASSETS` binding. The `/hf-transformers/` allowlist is **derived from
`MODEL_PRESETS` at module load** (every `transformers-js` preset id is split into owner/repo — a
catalog invariant guarded by `test/models.test.ts`); the `/hf/` (11 mlc-ai repos + 2 welcoma) and
`/gh-raw/` allowlists are explicit constants. The proxy forwards only `Accept`/`Range` headers
(never conditional headers — a 304 can make WebLLM's artifact cache treat a load as failed),
strips cookies/CSP headers, adds wildcard CORS + immutable caching, and stamps `X-Proxy-Worker`
on every proxy-route and diagnostics response (which `proxy-verify.ts` checks); static-asset
responses carry `X-App-Version` instead.

## Conventions checklist for changes

1. Verify against source before coding; this repo's types are strict and the loose Transformers.js
   surface is deliberately modeled structurally (`TransformersModule`, `MultimodalTransformers`).
2. Keep `.js` extensions on relative imports; keep `moduleResolution: "Bundler"`.
3. New public API must be exported through `src/index.ts` (values and types separately).
4. All thrown errors must be (or wrap into) `BrowserAIError` subclasses.
5. `onDelta`-style callbacks receive full accumulated text, not deltas.
6. Engine imports stay dynamic; kokoro-js stays a runtime import; don't touch `sideEffects`.
7. Model-cache code must only ever touch the named scopes listed above.
8. Keep the examples type-checking (`npx tsc -p examples/tsconfig.json` is a CI gate) and the
   worker-template compiling against the catalog.
9. `npm run build` and `npm test` must both pass before any release; `prepublishOnly` re-runs them.
10. Update this file when architecture-level facts change (counts, invariants, routes, configs).
