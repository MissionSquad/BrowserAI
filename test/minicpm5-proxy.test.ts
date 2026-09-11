import { afterEach, describe, expect, it, vi } from "vitest";
import worker from "../worker-template/cloudflare-worker.js";

const repos = ["RASMUS/MiniCPM5-2B-ONNX", "j4ys0n/MiniCPM5-2B-heretic-abliterated-ONNX"];
const env = { ASSETS: { fetch: async () => new Response("asset") } };

afterEach(() => vi.unstubAllGlobals());

describe("MiniCPM5 artifact proxy", () => {
  it("includes both variants in the catalog-derived allowlist", async () => {
    const response = await worker.fetch(new Request("https://example.com/__worker-health"), env);
    const health = await response.json() as { allowedHuggingFaceRepos: string[] };
    expect(health.allowedHuggingFaceRepos).toEqual(expect.arrayContaining(repos));
  });

  it.each(repos)("proxies the external weight file for %s", async (repo) => {
    const path = `${repo}/resolve/main/onnx/model_q4f16.onnx_data`;
    const fetchMock = vi.fn(async () => new Response(null, {
      headers: { "Content-Type": "application/octet-stream", "Content-Length": "1833893888" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const response = await worker.fetch(new Request(`https://example.com/hf-transformers/${path}`, { method: "HEAD" }), env);
    expect(response.status).toBe(200);
    expect(response.headers.get("X-Upstream-URL")).toBe(`https://huggingface.co/${path}`);
    expect(response.headers.get("Content-Length")).toBe("1833893888");
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
