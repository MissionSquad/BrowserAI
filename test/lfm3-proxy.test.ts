import { afterEach, expect, it, vi } from "vitest";
import worker from "../worker-template/cloudflare-worker.js";

const env = { ASSETS: { fetch: async () => new Response("asset") } };
const textId = "onnx-community/LFM3-1.7B-ONNX";
const visionId = "onnx-community/LFM3-VL-2B-ONNX";
afterEach(() => vi.unstubAllGlobals());

it("allows both LFM3 repositories through the hosted artifact proxy", async () => {
  const response = await worker.fetch(new Request("https://example.com/__worker-health"), env);
  const health = await response.json() as { allowedHuggingFaceRepos: string[] };
  expect(health.allowedHuggingFaceRepos).toEqual(expect.arrayContaining([textId, visionId]));
});

it.each([
  [textId, "onnx/model_q4f16.onnx_data"],
  [visionId, "onnx/decoder_model_merged_q4f16.onnx_data"],
  [visionId, "onnx/embed_tokens_fp16.onnx_data"],
  [visionId, "onnx/vision_encoder_fp16.onnx_data"],
  [visionId, "processor_config.json"],
])("proxies %s/%s without changing its artifact path", async (id, file) => {
  const path = `${id}/resolve/main/${file}`;
  const upstream = vi.fn(async () => new Response(null, { headers: { "Content-Type": "application/octet-stream", "Content-Length": "123" } }));
  vi.stubGlobal("fetch", upstream);
  const response = await worker.fetch(new Request(`https://example.com/hf-transformers/${path}`, { method: "HEAD" }), env);
  expect(response.status).toBe(200);
  expect(response.headers.get("X-Upstream-URL")).toBe(`https://huggingface.co/${path}`);
  expect(response.headers.get("Content-Length")).toBe("123");
  expect(upstream).toHaveBeenCalledOnce();
});
