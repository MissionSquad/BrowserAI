# MiniCPM5 2B browser integration

The regular model is selectable as `RASMUS/MiniCPM5-2B-ONNX` in both the catalog and text slot. It uses Transformers.js 4.2.0 with WebGPU and `q4f16`. No new JavaScript runtime dependency is needed. Its Hugging Face repository is automatically included in the existing Transformers artifact proxy allowlist and cache management.

The original model is [OpenBMB MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B); the browser artifact is [RASMUS's ONNX export](https://huggingface.co/RASMUS/MiniCPM5-2B-ONNX). It requires `shader-f16`, downloads approximately 1.83 GB of weights, and defaults to thinking disabled. Memory usage is estimated at 3 GB and varies with context. The existing default model is unchanged. JSON schemas remain prompt instructions on this backend; they are not grammar-constrained decoding.

## Runtime choice

A local Chrome 152 / macOS 26.4 Apple WebGPU comparison used the same system/user messages, temperature zero, thinking disabled and a 128-token output cap. Each runtime ran once to warm up and then three measured times. The [raw timing data](minicpm5-benchmark.json) records every run.

| Runtime / artifact | Mean warm decode | Mean warm first token | Weight download |
| --- | ---: | ---: | ---: |
| Transformers.js / RASMUS q4f16 ONNX | 55.64 tokens/s | 109 ms | 1.83 GB |
| WebLLM 0.2.84 / ozhyhinas q4f16_1 MLC | 46.21 tokens/s | 107 ms | 1.42 GB |

ONNX had about 20% higher decode throughput on this machine. WebLLM offers native prefill/decode/grammar telemetry and smaller downloads, but the requested priority was speed. Transformers.js already reports measured TTFT, token counts, elapsed time and token intervals in these projects. The prompt-count fix ensures the reported prompt includes the same thinking-template options used for generation.

This is a small hardware-specific sample, not a universal benchmark. The quantizations produced different outputs and completion lengths; total generation duration is not used to rank them. The MLC comparison used [this WebLLM 0.2.84-compatible artifact](https://huggingface.co/ozhyhinas/MiniCPM5-2B-q4f16_1-MLC), with a 4096-token context.

## Heretic conversion and remaining hosting step

The requested [Abiray GGUF card](https://huggingface.co/Abiray/MiniCPM5-2B-heretic-abliterated-GGUF) references [insraq's original Heretic safetensors checkpoint](https://huggingface.co/insraq/MiniCPM5-2B-heretic-abliterated). No published ONNX/MLC conversion of that fine-tune was found on September 10, 2026. The included script converts the original checkpoint, pinned at `8d83eaf42e30f4e4f164b8685be0ecdccd8ef0d8`, directly to ONNX. It never reads GGUF.

The converted model has been tested locally in both projects with streaming, thinking on/off, token/timing reporting, and SDK load/unload/cache inspection. Its 473 graph nodes and input/output shapes match the pinned regular ONNX graph. The conversion needs a public artifact repository before a permanent, loadable Heretic catalog entry can be added. Do not point a Transformers.js preset at the safetensors or GGUF repository: neither contains ONNX files.

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

After selecting an artifact repository, authenticate using `hf auth login` and upload the output folder with `hf upload OWNER/REPO /tmp/minicpm5-conversion/browser-model . --repo-type model`. Then add a separate `transformers-js` / `q4f16` preset using that exact `OWNER/REPO`, linking its `officialRepo` to the original insraq checkpoint. The regular and fine-tuned variants must retain distinct IDs, cache paths and history labels.
