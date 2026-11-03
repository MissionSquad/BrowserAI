import { afterEach, describe, expect, it, vi } from "vitest";
import { getModelPreset, modelSlots, presetModalityIn, presetModalityOut } from "../src/models.js";
import { vendorForModel } from "../src/history.js";
import worker from "../worker-template/cloudflare-worker.js";

const id = "j4ys0n/NVIDIA-Nemotron-3.5-Bolt-E4B-ONNX";
const env = { ASSETS: { fetch: async () => new Response("asset") } };

afterEach(() => vi.unstubAllGlobals());

describe("Nemotron 3.5 Bolt multimodal export", () => {
  it("routes the export through the multimodal runtime and capability slots", () => {
    const preset = getModelPreset(id);
    expect(preset.id).toBe(id);
    expect(preset.backend).toBe("transformers-js");
    expect(preset.transformersDtype).toBe("q4f16");
    expect(preset.mmRuntime).toBe("gemma-mm");
    expect(preset.disableThinkingSupported).toBe(false);
    expect(modelSlots(preset)).toEqual(["text", "vision", "stt"]);
    expect(presetModalityIn(preset)).toEqual(["text", "image", "audio"]);
    expect(presetModalityOut(preset)).toEqual(["text"]);
    expect(vendorForModel(preset)).toBe("nvidia");
    expect(preset.stability).toBe("experimental");
  });

  it("includes the export in the catalog-derived proxy allowlist", async () => {
    const response = await worker.fetch(new Request("https://example.com/__worker-health"), env);
    const health = await response.json() as { allowedHuggingFaceRepos: string[] };
    expect(health.allowedHuggingFaceRepos).toContain(id);
  });

  it.each([
    "onnx/decoder_model_merged_q4f16.onnx",
    "onnx/decoder_model_merged_q4f16.onnx_data",
    "onnx/decoder_model_merged_q4f16.onnx_data_1",
    "onnx/embed_tokens_q4f16.onnx",
    "onnx/embed_tokens_q4f16.onnx_data",
    "onnx/vision_encoder_q4f16.onnx",
    "onnx/vision_encoder_q4f16.onnx_data",
    "onnx/audio_encoder_q4f16.onnx",
    "onnx/audio_encoder_q4f16.onnx_data",
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
  ])("proxies the published artifact %s", async (file) => {
    const path = `${id}/resolve/main/${file}`;
    const fetchMock = vi.fn(async () => new Response(null, {
      headers: { "Content-Type": "application/octet-stream" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const response = await worker.fetch(new Request(`https://example.com/hf-transformers/${path}`, { method: "HEAD" }), env);
    expect(response.status).toBe(200);
    expect(response.headers.get("X-Upstream-URL")).toBe(`https://huggingface.co/${path}`);
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
