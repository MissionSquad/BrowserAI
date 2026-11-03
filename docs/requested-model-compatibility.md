# Requested model compatibility — 2026-11-03

Both projects use `@huggingface/transformers` 4.2.0 and `@mlc-ai/web-llm` 0.2.84. Assessment is against those installed versions, the public model cards, repository configurations, and file inventories. An ONNX export alone does not establish Transformers.js compatibility. Unsupported entries have not been added to the selectable catalog.

## Added: Nemotron 3.5 Bolt E4B

[Multimodal export](https://huggingface.co/j4ys0n/NVIDIA-Nemotron-3.5-Bolt-E4B-ONNX), built from [NVIDIA's BF16 source](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16), is public and configured in both catalogs. Verified revision: [`19e21d17a45b0f93808fa1d22ae6b4eb6ddabd9c`](https://huggingface.co/j4ys0n/NVIDIA-Nemotron-3.5-Bolt-E4B-ONNX/tree/19e21d17a45b0f93808fa1d22ae6b4eb6ddabd9c). All 21 hosted artifact files match the validated local checksums and are accessible without authentication.

- The experimental q4f16 preset accepts text, image, and audio and produces text, using the existing `gemma-mm` runtime and NVIDIA branding. Thinking cannot be enabled through this preset.
- The export contains decoder and token embedding graphs and weight shards, plus vision and audio encoder graphs and external weights. The nine ONNX files total approximately 5.21 GB; tokenizer and processor metadata are also required.
- Browser inference for text, image, and audio was smoke-tested on Apple WebGPU. Output quality and memory usage have not been benchmarked. The catalog's 8 GB GPU-memory figure is a planning estimate, not a measured minimum.
- The proxy allowlists derive from the catalog; regression checks cover capability routing, vendor attribution, and all nine ONNX graph and weight paths.

## Deferred pending a scope decision

| Model | Verified blocker | Work needed for browser-local support |
| --- | --- | --- |
| [LFM3-1.7B-ONNX](https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX) | `lfm3` / `Lfm3ForCausalLM` is absent from the installed Transformers.js model registry. | Transformers.js [4.3.2](https://github.com/huggingface/transformers.js/releases/tag/4.3.2) now registers LFM3. Upgrade and use the [Transformers.js-specific export](https://huggingface.co/onnx-community/LFM3-1.7B-ONNX) with the existing text pipeline, then validate browser inference. The originally requested LiquidAI export has a separate direct-ORT recipe. |
| [LFM3-VL-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-VL-2B-ONNX) | `lfm3_vl` model and processor support are missing. | Transformers.js 4.3.2 now registers LFM3-VL. Upgrade and use the [Transformers.js-specific export](https://huggingface.co/onnx-community/LFM3-VL-2B-ONNX), then adapt processor argument order (image before text) and validate inference. Its graph layout differs from the requested LiquidAI export. Video still requires an additional input API/UI. |
| [LFM3-Audio-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-Audio-2B-ONNX) | `lfm3_audio` is unsupported; existing STT/TTS runtimes do not implement its generation protocol. | Add 16 kHz log-mel preprocessing, audio feature insertion, speech-code feedback, and depth/codec decoding. Speech output requires 12 codebooks, three MaskGIT passes per frame, and inverse STFT to 24 kHz audio. The browser recipe pairs `decoder_q4f16` with float32 audio/codec side graphs and a float32 depth decoder; prefill chunks are at most 1,024 tokens. |
| [LFM3-Omni-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-Omni-2B-ONNX) | `lfm3_omni` is unsupported; it combines the missing vision and audio protocols. | Combine the VL and Audio adapters, then implement mixed input handling and interleaved text/speech output. Validate slot ownership and concurrent audio playback. The browser recipe uses `decoder_q4f16`, `vision_encoder_q8`, float32 audio/codec/depth graphs, and prefill chunks up to 1,024 tokens. Video requires the additional interface work above. |
| [Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) | `arktts` custom Python implementation; repository ships `model.safetensors` and `codec.pth`, with no ONNX artifacts. | Export and validate the slow/fast autoregressive networks and codec for WebGPU, quantize them, and implement the DualAR generation loop and 44.1 kHz audio handling in JS. Voice cloning also requires reference-audio/transcript inputs and codec encoding. Alternatively integrate its Python serving stack through a new remote backend. |
| [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) | `unlimited-ocr` custom Python implementation and safetensors only; no ONNX artifacts. Code includes custom sparse attention/Triton paths. | Export the vision/projector/decoder stack and provide browser-compatible sparse-attention operations, then port page/image preprocessing and output handling. Validate multi-page memory and numerical parity. A Python/server integration is a separate alternative to a substantial browser port. |
| [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | Gated `pyannote.audio` Python pipeline with PyTorch segmentation/embedding weights and PLDA assets, not a transcription model. | Obtain authorized model access; export segmentation and embedding models; implement windowing, speaker clustering/PLDA, overlap handling, and timestamp reconciliation. Add a diarization result API and speaker timeline UI. Alternatively add a server endpoint running pyannote. Access terms and authentication must be handled deliberately rather than relying on the existing anonymous artifact proxy. |

## Suggested order if further implementation is approved

1. Upgrade to Transformers.js 4.3.2 and integrate the compatible LFM3 text and vision exports; validate both in the browser.
2. Treat LFM3 Audio and Omni as separate runtime work; 4.3.2 does not register those architectures.
3. Decide whether Audio8, Unlimited OCR, and pyannote should remain browser-local. Server integration avoids browser export work but changes the projects' local-inference architecture and requires explicit product/API decisions.

No conversion jobs, new server endpoints, dependency upgrades, or unsupported selectable entries were introduced for deferred models.
