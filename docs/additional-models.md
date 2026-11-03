# Additional browser models

These four presets are available in BrowserAI and browserlab. Nemotron Bolt's separately published multimodal conversion is included in the same delivery; see its publication record.

| Preset | Runtime / precision | Approximate download | Input → output |
| --- | --- | --- | --- |
| `onnx-community/granite-docling-258M-ONNX` | Transformers.js, FP32 | 1.27 GB | Page image + instruction → DocTags |
| `onnx-community/nsfw_image_detection-ONNX` | Transformers.js, FP32 | 343 MB | Image → normal/nsfw scores |
| `enriqueag/phonon-2.5-onnx` | ORT Web 1.31.1, exact2x2 GPU encoder + int8 WASM decoder | 381 MB | 16 kHz mono audio → text |
| `istupakov/parakeet-tdt-0.6b-v3-onnx` | ORT Web 1.31.1, int8 encoder/decoder on WASM | 671 MB | 16 kHz mono audio → text |

All are experimental. Download figures are not memory requirements. Phonon requires WebGPU; Parakeet is CPU/WASM and does not require GPU memory. Inference and captured audio remain on the client; initial artifacts are downloaded from their model repositories or the configured proxy.

## Image and document APIs

```ts
import { BrowserAI } from "@missionsquad/browserai";

const ai = new BrowserAI();
await ai.load("onnx-community/nsfw_image_detection-ONNX");
const classification = await ai.classifyImage(imageUrl);
console.log(classification.scores); // [{ label: "normal", score: ... }, ...]

await ai.load("onnx-community/granite-docling-258M-ONNX"); // replaces the Image slot owner
const document = await ai.describeImage(imageUrl, "Convert this page to docling.");
console.log(document.text);
await ai.dispose();
```

The classifier uses the Image slot and returns scores with elapsed-time information. It does not accept an instruction prompt. Browserlab hides that prompt and displays both scores. These are model predictions, not definitive moderation decisions.

Docling has its own prompt, image splitting, and 4096-token limit. Its document/table/location tokens are marked special by the tokenizer, so the generic special-token stripping is disabled only for this preset. The final transport end-of-text marker is removed while DocTags remain intact. Browserlab renders this as literal text so document tags cannot be interpreted as HTML. Input is a page image; PDF rasterization, multi-page ingestion, and DocTags-to-Markdown conversion are not part of this change.

FP32 is deliberate. On Apple WebGPU, the classifier's FP16 and q4f16 outputs were near 50/50 on a benign cat image despite corresponding CPU models producing normal≈0.99989. Docling's q4f16 path emitted repeated punctuation. The chosen FP32 paths produced the expected classification and structured invoice output. The smaller exports remain unselected pending separate validation.

## Speech APIs and lifecycle

```ts
await ai.load("enriqueag/phonon-2.5-onnx");
const controller = new AbortController();
const result = await ai.transcribe(pcm16kMono, textSoFar => console.log(textSoFar), controller.signal);
console.log(result.text, result.segments);
```

TDT presets require decoded mono `Float32Array` PCM at 16 kHz. Use the existing `decodeAudioTo16kMono` helper for files/URLs. The third transcription argument is an optional AbortSignal; cancellation is cooperative between inference calls, not interruption of an in-flight WASM/GPU operation. Bounded macrotask yields permit UI cancellation events during long decode loops.

Audio longer than 30 seconds is split near quiet intervals in the last five seconds of each window. Segments report these chunk boundaries, not word alignment. Delta callbacks report the full accumulated transcript. Empty/very short clips may yield no text. Requests on a loaded model remain serialized; unloading waits for active inference and releases sessions.

The adapters use the published token-duration decoder contracts, preserve recurrent state on blank predictions, bound emissions at a frame, and release temporary tensors. Phonon's frontend is verified against an independent numerical reference. Parakeet's published `nemo128.onnx` graph cannot load in the browser WASM runtime because it uses an unsupported cast; the equivalent JavaScript frontend matches that graph's CPU output within `1e-4` on the recorded fixture, including reflect padding and the final frame.

## Runtime, cache and proxy behavior

The `onnxruntime-tdt` npm alias pins ONNX Runtime Web 1.31.1 independently of Transformers.js 4.2.0's existing runtime. The new native WebGPU bundle requires matching **asyncify** MJS/WASM assets; JSEP assets cannot initialize it. Defaults reference the pinned official jsDelivr distribution. SDK applications can override `new BrowserAI({ tdt: { wasmPaths: { mjs: "/ort/ort-wasm-simd-threaded.asyncify.mjs", wasm: "/ort/ort-wasm-simd-threaded.asyncify.wasm" } } })` when serving those matching assets themselves.

TDT artifacts use the existing `transformers-cache` bucket and each preset's own repository URL, so existing cache inspection/deletion remains scoped to that model. Storage open/read/write failures emit `cache-warning` and inference continues without persistent caching. Failed HTTP downloads remain errors. The worker derives allowlists from both Transformers.js and ONNX Runtime presets. History and runtime labels preserve the new `onnx-runtime` backend.

## Verification

See [recorded browser results](./additional-models-validation.json). In an isolated headless Chrome instance on Apple WebGPU:

- Docling extracted the invoice number, customer name, and total with DocTags intact.
- The classifier assigned normal≈0.999887 to the benign image.
- Phonon 2.5 and Parakeet TDT both correctly transcribed the public 11-second JFK clip.

These are single-case smoke checks using local copies of the published artifacts, not broad accuracy, latency, multilingual, or cross-browser benchmarks. Unit coverage includes DSP reference fixtures, both decoder contracts, recurrent state handling, cancellation, failed downloads/cache writes, partial-load cleanup, chunk conservation, model slots, proxy paths, and backend history.

## Sources and attribution

- [IBM Granite Docling export](https://huggingface.co/onnx-community/granite-docling-258M-ONNX), Apache-2.0.
- [Falconsai classifier](https://huggingface.co/Falconsai/nsfw_image_detection) and [ONNX export](https://huggingface.co/onnx-community/nsfw_image_detection-ONNX), original model Apache-2.0.
- [Fermion Phonon 2.5](https://huggingface.co/FermionResearch/Phonon-2.5) and [enriqueag ONNX export](https://huggingface.co/enriqueag/phonon-2.5-onnx), model weights CC-BY-4.0. The runtime follows the exporter's published preprocessing and TDT contracts.
- [NVIDIA Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [Ivan Stupakov ONNX export](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx), model weights CC-BY-4.0. The frontend's small sparse mel matrix is derived from the credited export; its graph SHA-256 is recorded in the implementation and fixture.
