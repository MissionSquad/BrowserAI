"""Export the original MiniCPM5 Heretic safetensors checkpoint for Transformers.js.

See docs/minicpm5.md for pinned dependencies and usage. This script never uploads.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from huggingface_hub import snapshot_download
import onnx
import onnxruntime_genai

SOURCE_REPO = "insraq/MiniCPM5-2B-heretic-abliterated"
SOURCE_REVISION = "8d83eaf42e30f4e4f164b8685be0ecdccd8ef0d8"
REFERENCE_REPO = "RASMUS/MiniCPM5-2B-ONNX"
REFERENCE_REVISION = "3f3d18bc4cc0a82bdb5dcf6f3c5894b6c350290c"


def prepare_browser_model(source: Path, exported: Path, output: Path) -> dict:
    config = json.loads((source / "config.json").read_text())
    assert config["model_type"] == "llama" and config["num_hidden_layers"] == 42
    assert config["head_dim"] == 128
    model = onnx.load(exported / "model.onnx", load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)
    assert ops["GroupQueryAttention"] == 42 and ops["MatMulNBits"] == 211
    assert not any(ops[name] for name in ["MultiHeadAttention", "Loop", "Scan"])

    # Transformers.js cannot resolve the exporter's symbolic KV head dimension.
    patched = 0
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "kv_cache_dim":
                dim.dim_value = config["head_dim"]
                patched += 1
    assert patched >= 84
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                assert entry.value == "model.onnx.data"
                entry.value = "model_q4f16.onnx_data"

    # This graph uses ORT's SimplifiedLayerNormalization in the default domain,
    # which the generic ONNX checker does not register. Compare its operators,
    # attributes and I/O to the pinned, browser-tested regular export instead.
    reference_url = (
        f"https://huggingface.co/{REFERENCE_REPO}/resolve/{REFERENCE_REVISION}"
        "/onnx/model_q4f16.onnx"
    )
    with urllib.request.urlopen(reference_url, timeout=120) as response:
        reference = onnx.load_model_from_string(response.read())
    assert model.ir_version == reference.ir_version
    assert model.opset_import == reference.opset_import
    assert model.graph.node == reference.graph.node
    assert model.graph.input == reference.graph.input
    assert model.graph.output == reference.graph.output
    assert [(t.name, list(t.dims), t.data_type) for t in model.graph.initializer] == [
        (t.name, list(t.dims), t.data_type) for t in reference.graph.initializer
    ]

    (output / "onnx").mkdir(parents=True, exist_ok=True)
    (output / "onnx/model_q4f16.onnx").write_bytes(model.SerializeToString())
    shutil.copyfile(exported / "model.onnx.data", output / "onnx/model_q4f16.onnx_data")
    config["transformers.js_config"] = {
        "kv_cache_dtype": {"q4f16": "float16"},
        "use_external_data_format": {"model_q4f16.onnx": 1},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    template = (source / "chat_template.jinja").read_text()
    # This unused assignment requires a min filter absent from @huggingface/jinja.
    dead_line = "            {%- set min_count = [tool_calls_count, tool_sep_count]|min %}\n"
    assert template.count("min_count") == 1 and dead_line in template
    template = template.replace(dead_line, "")
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
    tokenizer_config["chat_template"] = template
    (output / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n")
    (output / "chat_template.jinja").write_text(template)
    for name in ["tokenizer.json", "generation_config.json"]:
        shutil.copyfile(source / name, output / name)
    (output / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: insraq/MiniCPM5-2B-heretic-abliterated\n"
        "library_name: transformers.js\npipeline_tag: text-generation\n"
        "tags: [onnx, webgpu, minicpm5, heretic]\nlanguage: [en, zh]\n---\n\n"
        "# MiniCPM5 2B Heretic — ONNX q4f16\n\n"
        f"Converted from [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO}) "
        f"at revision `{SOURCE_REVISION}`. Original model: OpenBMB MiniCPM5-2B. "
        "This is the original safetensors checkpoint referenced by Abiray's GGUF model card; "
        "no GGUF files are used. Conversion changes precision, not the fine-tuning method.\n\n"
        "Load with Transformers.js 4.2.0, `device: 'webgpu'`, `dtype: 'q4f16'`. "
        "Requires shader-f16; weights are approximately 1.83 GB. "
        "Use `tokenizer_encode_kwargs: { enable_thinking: false }` to disable thinking.\n\n"
        "See conversion-manifest.json for source revision, tool versions and file checksums.\n"
    )

    manifest = {
        "source": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "reference_graph": reference_url,
        "quantization": "int4 weights / float16 I/O and KV cache (q4f16)",
        "operators": dict(ops),
        "kv_dimensions_patched": patched,
        "tool_versions": {name: version(name) for name in [
            "onnxruntime-genai", "onnxruntime", "onnx", "onnx-ir", "onnxscript", "torch", "transformers", "numpy",
        ]},
        "files": {},
    }
    for file in sorted(output.rglob("*")):
        if file.is_file() and file.name != "conversion-manifest.json":
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["files"][str(file.relative_to(output))] = {"bytes": file.stat().st_size, "sha256": digest}
    (output / "conversion-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Existing pinned source snapshot; otherwise download it.")
    parser.add_argument("--package-only", action="store_true", help="Package an existing work-dir/exported graph.")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = args.source_dir.resolve() if args.source_dir else Path(snapshot_download(
        SOURCE_REPO, revision=SOURCE_REVISION, local_dir=work / "source",
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    ))
    if not args.package_only:
        export_input = work / "export-input"
        export_input.mkdir(exist_ok=True)
        for file in source.iterdir():
            if file.is_file() and file.suffix in [".json", ".jinja", ".safetensors"]:
                target = export_input / file.name
                if not target.exists():
                    target.symlink_to(file)
        # The source was saved by Python Transformers 5. Export with 4.57.6 using
        # its equivalent generic fast tokenizer, without changing the source files.
        target = export_input / "tokenizer_config.json"
        target.unlink()
        tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
        tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
        target.write_text(json.dumps(tokenizer_config))
        builder = Path(onnxruntime_genai.__file__).parent / "models/builder.py"
        subprocess.run([
            sys.executable, str(builder), "-i", str(export_input), "-o", str(work / "exported"),
            "-p", "int4", "-e", "webgpu", "-c", str(work / "cache"),
        ], check=True)
    manifest = prepare_browser_model(source, work / "exported", work / "browser-model")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
