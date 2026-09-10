# @missionsquad/browserai

Run small open AI models **entirely in the browser** with WebGPU — no inference server — as an
embeddable, UI-free TypeScript SDK. BrowserAI packages a curated 47-model catalog, model
loading/downloading with progress, capability-slot management, three inference runtimes, streaming,
schema-constrained JSON generation, run metrics, browser-cache management, and a complete
press-to-talk voice-assistant pipeline. Drop it into any frontend site and build your own UI on top.

Three inference runtimes:

- **WebLLM (MLC)** for quantized MLC text models, with schema-constrained JSON decoding and detailed
  per-token runtime telemetry.
- **Transformers.js + ONNX/WebGPU** for browser-ready ONNX models: text LLMs, Whisper/Moonshine/
  Parakeet transcription, Supertonic speech, SmolVLM/Qwen-VL/OCR vision, Gemma 4 multimodal, and
  audio-LLM ASR (Granite Speech, Voxtral realtime).
- **kokoro-js** (loaded from its CDN bundle by default) for Kokoro 82M text-to-speech, including
  sentence-by-sentence streaming synthesis.

## Install

```bash
npm install @missionsquad/browserai
```

Requirements: a WebGPU-capable browser (lightweight speech models — Whisper/Moonshine/Parakeet
transcription and the TTS engines — can fall back to wasm; text, vision, and audio-LLM models
require WebGPU), Node.js 20+ for development. `@mlc-ai/web-llm` and `@huggingface/transformers` are regular
dependencies, dynamically imported only when a model that needs them loads — your initial bundle
stays small.

## Quickstart

```ts
import { BrowserAI } from "@missionsquad/browserai";

const ai = new BrowserAI();
ai.on("loadprogress", ({ progress, status }) => console.log(progress, status));

await ai.load("Qwen3.5-0.8B-q4f16_1-MLC"); // downloads once, then serves from browser cache

const { text, stats } = await ai.generateText(
  [
    { role: "system", content: "You are terse." },
    { role: "user", content: "Why is the sky blue?" },
  ],
  { runtime: { jsonMode: "none" }, onDelta: (full) => render(full) },
);
console.log(text, stats.extra?.decode_tokens_per_s);
```

### Capability slots

Models load into one of four **capability slots** — `text`, `vision`, `stt` (transcription), `tts`
(speech) — with at most one model per slot, so up to four models can be resident at once. Loading a
model whose slots are occupied replaces the conflicting owner. Loads/unloads are single-flight, and
runs are serialized per loaded model, so concurrent calls queue instead of driving one engine from
two places.

```ts
await ai.load("onnx-community/whisper-base");          // stt slot
await ai.load("onnx-community/Kokoro-82M-v1.0-ONNX");  // tts slot

const { text } = await ai.transcribe(pcm16k);           // Float32Array @ 16 kHz mono
const clip = await ai.synthesizeToClip("Hello!", { voice: "af_heart" });
ai.occupiedSlots();                                     // Set { "text", "stt", "tts" }
```

### Schema-constrained JSON

```ts
import { buildExtractionMessages, parseJsonFromModel, validateAgainstJsonSchema } from "@missionsquad/browserai";

const messages = buildExtractionMessages(inputText, mySchema);
const { text } = await ai.generateText(messages, { schema: mySchema }); // WebLLM: real constrained decoding
const parsed = parseJsonFromModel(text);
if (parsed.ok) console.log(validateAgainstJsonSchema(parsed.value, mySchema)); // [] when valid
```

WebLLM backends enforce the schema with a grammar-constrained decoder; Transformers.js backends are
prompt-guided, so validate the output either way.

### Voice pipeline (press-to-talk assistant)

`VoicePipeline` chains mic → STT → text LLM → TTS → speakers as a headless state machine with typed
events. With Kokoro in the speech slot, replies are spoken **while the text model is still
generating** (audio starts after the first sentence); other TTS models speak the finished reply as
one clip. Pressing talk during playback stops it (barge-in).

```ts
import { BrowserAI, VoicePipeline } from "@missionsquad/browserai";

const ai = new BrowserAI();
const voice = new VoicePipeline(ai);

await voice.loadRecommendedModels(); // Whisper Base + Qwen3.5 0.8B + Kokoro 82M (~2.5 GB VRAM)

voice.on("stagechange", ({ stage }) => setIndicator(stage)); // idle/listening/transcribing/thinking/speaking/error
voice.on("transcript", ({ text }) => addUserBubble(text));
voice.on("replydelta", ({ text }) => updateAssistantBubble(text));
voice.on("turncommitted", ({ messages }) => renderHistory(messages));
voice.on("error", ({ label, message }) => showError(`${label}: ${message}`));

talkButton.onclick = () => voice.toggle(); // press to talk, press again to send
newChatButton.onclick = () => voice.reset();
```

The turn commits to history only on success; discarded takes (`turndiscarded`: too short, no speech,
empty reply, contention with other work) leave history untouched. For input-level visualization,
subscribe to `micstream` (attach an AnalyserNode) and `micstreamended`; for output, pass
`streamingAudio.onGraphReady` to tap the streaming player's gain node.

### Vision

```ts
await ai.load("HuggingFaceTB/SmolVLM-256M-Instruct");
const answer = await ai.describeImage(imageDataUrl, "What does this chart show?", (partial) => render(partial));
```

### Metrics + run history

Every generation returns `RuntimeStats` (tokens, decode tok/s, TTFT, per-token latency breakdown for
WebLLM, measured telemetry for Transformers.js). Persist and aggregate runs with `RunHistoryStore`:

```ts
import { RunHistoryStore, runRecordFromStats } from "@missionsquad/browserai";

const history = new RunHistoryStore(); // localStorage-backed; pass { storage: null } for in-memory
const record = runRecordFromStats("chat", stats);
if (record) history.append(record);
history.aggregateByModel("chat"); // decode/ttft averages per model
```

### Cache management

Models download once (hundreds of MB to a few GB) and are cached in browser storage — WebLLM weights
in IndexedDB by default (configurable via `cacheBackend`), Transformers.js artifacts always in the
Cache API bucket `transformers-cache`.

```ts
await ai.cacheStatus("onnx-community/whisper-base"); // { downloaded, matchedEntries, … }
await ai.deleteModelArtifacts("onnx-community/whisper-base");
await ai.deleteAllModelArtifacts();                  // releases loaded models first
await ai.estimateStorage();                          // origin-wide usage/quota
```

Cleanup targets model artifact storage only — never cookies, localStorage, or unrelated site data.

## Configuration

```ts
const ai = new BrowserAI({
  modelSource: "direct",          // "direct" (default, straight from Hugging Face) | "proxy"
  proxyOrigin: undefined,          // proxy Worker origin; defaults to the page origin in proxy mode
  verifyProxy: true,               // probe the proxy before each proxied load
  cacheBackend: "indexeddb",       // WebLLM artifact cache: "indexeddb" | "cache" | "opfs" | "cross-origin"
  webllm: {
    worker: () => new Worker(new URL("./llm.worker.ts", import.meta.url), { type: "module" }),
    logLevel: "INFO",
  },
  kokoro: { url: "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm" }, // or load: () => import("kokoro-js")
  tts: { voice: "af_heart", speed: 1, pitch: 1 },
  history: { key: "browserai:v1:history", maxRecords: 200 },
});
```

### Running WebLLM in a worker

By default WebLLM engines run on the main thread. To keep loading and token generation off it,
create a two-line worker entry in your app and pass its factory:

```ts
// llm.worker.ts — bundled by YOUR build (Vite/webpack understand this pattern)
export * from "@missionsquad/browserai/worker";
```

```ts
const ai = new BrowserAI({
  webllm: { worker: () => new Worker(new URL("./llm.worker.ts", import.meta.url), { type: "module" }) },
});
```

### Model-artifact proxy (optional)

Direct downloads work on any site. If a hosted deployment hits CORS or cache-storage edge cases with
cross-origin model downloads, deploy the Cloudflare Worker in [`worker-template/`](worker-template/)
and configure `{ modelSource: "proxy" }`. The Worker proxies `/hf/*` (WebLLM weights),
`/hf-transformers/*` (Transformers.js artifacts), and `/gh-raw/*` (WebLLM wasm libraries), restricted
to an allowlist derived from the SDK's model catalog. Kokoro is the one exception: kokoro-js ships
its own runtime and fetches its weights directly.

## Model catalog

47 presets across four capability types (see `MODEL_PRESETS`; each entry carries GPU-memory
estimates, download size, stability rating, and per-model default runtime parameters):

- **Text (WebLLM/MLC):** Qwen3.5 0.8B/2B/4B, Gemma 3 1B, Gemma 4 E2B/E4B (experimental custom
  records), Llama 3.2 1B/3B, Hermes 3, OLMo 2, Phi-4 mini, Ministral 3, Mistral 7B.
- **Text (Transformers.js/ONNX):** Nemotron 3 Nano, LFM2.5 230M/350M/1.2B-Thinking, Falcon H1 Tiny,
  the Granite 4.0 family, MiniCPM5, SmolLM3, Gemma 4 E2B/E4B ONNX (multimodal: text + image + audio).
- **Transcription:** Whisper Base/Tiny/Medium/Medium-EN, Moonshine Base/Tiny, Cohere Transcribe,
  Parakeet CTC 0.6B, Granite Speech 2B, Voxtral Mini 4B Realtime.
- **Speech:** Kokoro 82M (streams), Supertonic 1/2.
- **Vision:** SmolVLM 256M/500M, Qwen3-VL 2B, Qwen2.5-VL 3B, GLM-OCR, LightOnOCR-2.

## Browser constraints worth knowing

- **WebGPU** is required for text, vision, Gemma-multimodal, and audio-LLM models; ASR-pipeline
  (Whisper/Moonshine/Parakeet) and TTS presets fall back to wasm. `probeHardware()` reports
  adapter/features/limits; other loads throw `WebGPUUnavailableError` without it.
- `decodeAudioTo16kMono`, `capSrc` (image downscaling), `MicRecorder`, and the audio players use
  main-thread browser APIs (`AudioContext`, `Image`, canvas, `MediaRecorder`) — decode/capture on
  the page even if you move inference elsewhere.
- If you attach an analyzer to an `<audio>` element playing clips, remember a media element allows
  exactly **one** `MediaElementSourceNode` for its whole life — create the analyzer once and never
  close its context while the element is in use.
- All SDK errors are `BrowserAIError` subclasses (`ModelLoadError`, `MicrophoneError`,
  `MissingModelError`, `VoiceTurnError`, …) — branch with `instanceof`.

## Development

```bash
npm ci
npm run build   # type-check + emit dist (ESM + .d.ts)
npm test        # vitest unit tests
```

## License

ISC

### MiniCPM5 2B

MiniCPM5 2B is available through Transformers.js/WebGPU (`RASMUS/MiniCPM5-2B-ONNX`, q4f16). See [runtime measurements and the Heretic ONNX conversion recipe](docs/minicpm5.md). The Heretic conversion is tested locally; its catalog registration is pending an artifact hosting repository.
