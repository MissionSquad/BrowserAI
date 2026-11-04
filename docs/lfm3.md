# LFM3 with Transformers.js 4.3.2

Both projects pin `@huggingface/transformers` to **4.3.2** and include these experimental presets under the **Liquid** vendor:

| Preset | Project capability | Precision | Approximate graph download |
| --- | --- | --- | --- |
| `onnx-community/LFM3-1.7B-ONNX` | Text generation | q4f16 | 1.09 GB |
| `onnx-community/LFM3-VL-2B-ONNX` | Image question answering | FP16 vision/embeddings, q4f16 decoder | 1.94 GB |

Tokenizer/configuration files add to these sizes. GPU memory estimates in the catalog are planning estimates, not measured minimums. These paths require WebGPU with the relevant FP16 support.

The selected exports are explicitly packaged for Transformers.js. They come from the requested [LiquidAI text model](https://huggingface.co/LiquidAI/LFM3-1.7B) and [vision model](https://huggingface.co/LiquidAI/LFM3-VL-2B). The separately published `LiquidAI/*-ONNX` repositories use direct-ORT conventions and are not interchangeable with these artifact layouts.

## Runtime behavior

Text uses the existing text-generation pipeline, streaming and token accounting. Thinking is disabled by default and can be enabled using `runtime.disableThinking: false`. Its repetition penalty defaults to 1: in browser checks the shared 1.08 penalty changed a simple arithmetic response to an incorrect answer; the model-specific default produced the correct result. Other models' defaults are unchanged.

Vision uses `AutoModelForImageTextToText` with the export's mixed precision recipe. `Lfm3VlProcessor` takes **image before text**. The chat template already supplies special tokens, so preprocessing uses `add_special_tokens: false`. The template disables thinking, and generation uses temperature 0.1, top-k 50 and repetition penalty 1.1 as in the export's example. These options are scoped to the new preset; existing text-first VLMs, SmolVLM's mixed precision and Docling's FP32/DocTags behavior remain unchanged.

LFM3-VL occupies the Image slot. This integration exposes page/image inputs with text prompts and text answers; it does not add a video API or a text-only generation path through this VLM. Use the separate LFM3 text preset for chat and extraction.

## BrowserAI usage

```ts
import { BrowserAI } from "@missionsquad/browserai";

const ai = new BrowserAI();
await ai.load("onnx-community/LFM3-1.7B-ONNX");
const answer = await ai.generateText(
  [{ role: "user", content: "What is the capital of France?" }],
  { onDelta: textSoFar => console.log(textSoFar) },
);

await ai.load("onnx-community/LFM3-VL-2B-ONNX");
const imageAnswer = await ai.describeImage(imageUrl, "What is in this image?");
console.log(answer.text, imageAnswer.text);
await ai.unloadAll();
```

In browserlab, select Liquid in the Library vendor filter. Both entries can be loaded from their cards; use Chat/Text→JSON for LFM3 1.7B and Vision for LFM3-VL 2B.

## Validation and upgrade compatibility

[Recorded validation](./lfm3-validation.json) includes real isolated Chrome/Apple WebGPU inference using BrowserAI's public API and browserlab's actual Library, Chat and Vision controls. Tests used local copies of public model artifacts with the matching CDN runtime, not mocked inference.

- Text: streamed responses, repeated generation, arithmetic with the selected default, a factual question, an exact-word instruction, and thinking enabled/disabled.
- Vision: image-first preprocessing and counting the two cats in a public sample image. Browserlab now displays a bare numeric response such as `2.` as text rather than an empty Markdown list.
- Upgrade regression checks: FP32 image classifier, structured Docling extraction, and multimodal Bolt text/repeated text/image/audio inference.
- Unit tests cover tokenizer options and counts, per-model processor order/sampling, existing VLM behavior, and the hosted proxy's new model paths. Full SDK validation includes the examples type-check.
- Browserlab's production build still emits no redundant ONNX Runtime WASM assets and passes the 25 MiB asset guard and Wrangler deployment dry run.

The upgrade changes Transformers.js's ONNX Runtime Web dependency to `1.31.0-dev.20260914-8d85527a0`. Its matching assets remain version-specific CDN resources. The independent `onnxruntime-tdt` alias remains at 1.31.1 for Phonon/Parakeet. Do not mix these runtime binaries.

These are smoke checks, not a broad accuracy, memory, performance or cross-browser benchmark. An initially incorrect non-thinking arithmetic answer and differences from native CPU logits were investigated; identical prompt tokens were confirmed, but no general CPU/WebGPU numerical-equivalence claim is made. Both new entries remain experimental.

## Still deferred

LFM3 Audio and Omni architectures are not registered in Transformers.js 4.3.2. This upgrade adds the text and vision models only; speech generation, multimodal audio processing and video require separate runtime work.

Sources: [4.3.2 release](https://github.com/huggingface/transformers.js/releases/tag/4.3.2), [text export](https://huggingface.co/onnx-community/LFM3-1.7B-ONNX), [vision export](https://huggingface.co/onnx-community/LFM3-VL-2B-ONNX).
