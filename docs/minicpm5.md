# MiniCPM5 2B browser integration

Both variants use Transformers.js 4.2.0 with WebGPU and `q4f16`, and occupy the text slot. No new JavaScript runtime dependency is needed. Their separate Hugging Face repositories are automatically included in the existing Transformers artifact proxy allowlist and cache management.

| Variant | Catalog ID | Browser artifact |
| --- | --- | --- |
| MiniCPM5 2B | `RASMUS/MiniCPM5-2B-ONNX` | [Regular ONNX](https://huggingface.co/RASMUS/MiniCPM5-2B-ONNX) |
| MiniCPM5 2B Heretic | `j4ys0n/MiniCPM5-2B-heretic-abliterated-ONNX` | [Heretic ONNX](https://huggingface.co/j4ys0n/MiniCPM5-2B-heretic-abliterated-ONNX) |

The original model is [OpenBMB MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B); the browser artifact is [RASMUS's ONNX export](https://huggingface.co/RASMUS/MiniCPM5-2B-ONNX). It requires `shader-f16`, downloads approximately 1.83 GB of weights, and defaults to thinking disabled. Memory usage is estimated at 3 GB and varies with context. The existing default model is unchanged. JSON schemas remain prompt instructions on this backend; they are not grammar-constrained decoding.

## Runtime choice

A local Chrome 152 / macOS 26.4 Apple WebGPU comparison used the same system/user messages, temperature zero, thinking disabled and a 128-token output cap. Each runtime ran once to warm up and then three measured times. The [raw timing data](minicpm5-benchmark.json) records every run.

| Runtime / artifact | Mean warm decode | Mean warm first token | Weight download |
| --- | ---: | ---: | ---: |
| Transformers.js / RASMUS q4f16 ONNX | 55.64 tokens/s | 109 ms | 1.83 GB |
| WebLLM 0.2.84 / ozhyhinas q4f16_1 MLC | 46.21 tokens/s | 107 ms | 1.42 GB |

ONNX had about 20% higher decode throughput on this machine. WebLLM offers native prefill/decode/grammar telemetry and smaller downloads, but the requested priority was speed. Transformers.js already reports measured TTFT, token counts, elapsed time and token intervals in these projects. The prompt-count fix ensures the reported prompt includes the same thinking-template options used for generation.

This is a small hardware-specific sample, not a universal benchmark. The quantizations produced different outputs and completion lengths; total generation duration is not used to rank them. The MLC comparison used [this WebLLM 0.2.84-compatible artifact](https://huggingface.co/ozhyhinas/MiniCPM5-2B-q4f16_1-MLC), with a 4096-token context.

## Heretic conversion

The requested [Abiray GGUF card](https://huggingface.co/Abiray/MiniCPM5-2B-heretic-abliterated-GGUF) references [insraq's original Heretic safetensors checkpoint](https://huggingface.co/insraq/MiniCPM5-2B-heretic-abliterated). No published ONNX/MLC conversion of that fine-tune was found on September 10, 2026. The included script converts the original checkpoint, pinned at `8d83eaf42e30f4e4f164b8685be0ecdccd8ef0d8`, directly to ONNX. It never reads GGUF.

The converted model has been tested locally in both projects with streaming, thinking on/off, token/timing reporting, and SDK load/unload/cache inspection. Its 473 graph nodes and input/output shapes match the pinned regular ONNX graph. The converted files are published in the public, ungated [j4ys0n/MiniCPM5-2B-heretic-abliterated-ONNX](https://huggingface.co/j4ys0n/MiniCPM5-2B-heretic-abliterated-ONNX) repository. The catalog links to this ONNX artifact and attributes the fine-tune to the original insraq checkpoint. Both variants retain distinct IDs, cache paths and history labels.

To reproduce with Python 3.13 in an isolated environment:

```sh
python3 -m venv /tmp/minicpm5-convert-env
/tmp/minicpm5-convert-env/bin/pip install \
  onnxruntime-genai==0.15.2 onnxruntime==1.30.0 onnx==1.22.0 \
  onnx-ir==1.0.0 onnxscript==0.7.2 torch==2.14.0 \
  transformers==4.57.6 numpy==2.2.6 huggingface-hub==0.36.2
/tmp/minicpm5-convert-env/bin/python scripts/convert-minicpm5-heretic.py \
  --work-dir /tmp/minicpm5-conversion
```

Allow space for the approximately 5 GB source, intermediate export and 1.83 GB final weights. The output directory is `/tmp/minicpm5-conversion/browser-model`. It includes a model card, source revision, exact tool versions and SHA-256 checksums. The script applies the three browser fixes documented by RASMUS: concrete KV head dimension 128, an embedded text chat template, and removal of the unused unsupported Jinja `min` assignment. It also declares external data and an fp16 KV cache in `config.json`.

The conversion script prepares files locally and never uploads them. The public repository includes `conversion-manifest.json` with the source revision, tool versions and artifact checksums. Published ONNX weights have SHA-256 `66ed0b1b1e016e6cf65d1014aa6b542497333860ef561402b18103dbd39eae1a`.
