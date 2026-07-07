# BrowserAI examples

Five self-contained, runnable browser demos of [`@missionsquad/browserai`](../README.md). Each page
exercises one part of the SDK and is heavily commented so it doubles as documentation.

| Page | File | Demonstrates |
| --- | --- | --- |
| Text generation | [`text-generation.ts`](text-generation.ts) | `load`, `generateText`, streaming `onDelta`, `RuntimeStats` |
| Structured JSON | [`json-extraction.ts`](json-extraction.ts) | `buildExtractionMessages`, schema-constrained decoding, `parseJsonFromModel`, `validateAgainstJsonSchema` |
| Speech in & out | [`speech.ts`](speech.ts) | `MicRecorder`, `transcribe`, `synthesizeToClip`, `SynthClip` |
| Voice assistant | [`voice-assistant.ts`](voice-assistant.ts) | `VoicePipeline` (mic → STT → LLM → TTS) and its events |
| Vision Q&A | [`vision.ts`](vision.ts) | `describeImage` with a vision-language model |

## Run them

From this `examples/` directory:

```bash
npm install
npm run dev
```

Open the printed URL (e.g. `http://localhost:5173`) and pick an example from the landing page.

You do **not** need to build the SDK first. [`vite.config.ts`](vite.config.ts) aliases
`@missionsquad/browserai` to the local `../src/index.ts`, exactly like the test config does, so the
demos run against the source in this repo. To exercise the published build instead, change that alias
to `../dist/index.js` and run `npm run build` in the parent first.

## What to expect

- **A WebGPU-capable browser is required.** Recent Chrome, Edge, or another Chromium-based browser on
  a machine with a supported GPU. Text, vision, and multimodal models require WebGPU; the
  transcription and speech models fall back to wasm, so the Speech page works on more devices.
- **First load downloads the model** — anywhere from ~50 MB (Whisper Tiny, SmolVLM 256M) to a few GB
  (larger text models). The browser caches artifacts afterward (WebLLM weights in IndexedDB,
  Transformers.js artifacts in the `transformers-cache` Cache API bucket), so subsequent loads are
  fast. Watch the progress bar and the log panel on each page.
- **The microphone examples need permission.** The Speech and Voice pages call `getUserMedia`; allow
  microphone access when prompted. Denials surface as a typed `MicrophoneError`
  (`permissionDenied: true`).
- **Models download from Hugging Face directly** by default (`modelSource: "direct"`). If your
  environment needs a proxy, deploy the Worker in [`../worker-template/`](../worker-template/) and
  construct the client with `{ modelSource: "proxy", proxyOrigin }` — see the main README.

## Type-checking

The example modules are type-checked against the SDK's real types:

```bash
# from the repo root
npx tsc -p examples/tsconfig.json
```

(`vite.config.ts` is excluded from that check — it depends on `vite`, an examples-only devDependency.)
