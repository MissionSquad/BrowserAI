import { describe, expect, it } from "vitest";
import worker from "../worker-template/cloudflare-worker.js";

const ENV = {
  ASSETS: {
    fetch: async () => new Response("asset", { status: 200 }),
  },
};

describe("worker template proxy routes", () => {
  it("serves a health endpoint listing the derived allowlist", async () => {
    const response = await worker.fetch(new Request("https://example.com/__worker-health"), ENV);
    expect(response.status).toBe(200);
    expect(response.headers.get("X-Proxy-Worker")).toContain("browserai-proxy/");
    const body = (await response.json()) as { ok: boolean; allowedHuggingFaceRepos: string[] };
    expect(body.ok).toBe(true);
    // Derived entries: every Transformers.js preset, including the multimodal ones.
    expect(body.allowedHuggingFaceRepos).toContain("onnx-community/whisper-base");
    expect(body.allowedHuggingFaceRepos).toContain("onnx-community/moonshine-base-ONNX");
    expect(body.allowedHuggingFaceRepos).toContain("onnx-community/Kokoro-82M-v1.0-ONNX");
    // Explicit MLC entries.
    expect(body.allowedHuggingFaceRepos).toContain("mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC");
  });

  it("proxies an allowlisted /hf/ path to the Hugging Face upstream", async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = (async (input: RequestInfo | URL) => {
        const url = input instanceof Request ? input.url : String(input);
        expect(url).toBe("https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC/resolve/main/mlc-chat-config.json");
        return new Response('{"version":"0.1.0"}', { status: 200, headers: { "Content-Type": "application/json" } });
      }) as typeof fetch;

      const response = await worker.fetch(
        new Request("https://proxy.example.com/hf/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC/resolve/main/mlc-chat-config.json"),
        ENV,
      );

      expect(response.status).toBe(200);
      expect(response.headers.get("X-Proxy-Route")).toBe("hf");
      expect(response.headers.get("X-Upstream-URL")).toBe(
        "https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC/resolve/main/mlc-chat-config.json",
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("proxies an allowlisted /hf-transformers/ path (catalog-derived entry)", async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = (async (input: RequestInfo | URL) => {
        const url = input instanceof Request ? input.url : String(input);
        expect(url).toBe("https://huggingface.co/onnx-community/moonshine-base-ONNX/resolve/main/config.json");
        return new Response('{"model_type":"moonshine"}', { status: 200, headers: { "Content-Type": "application/json" } });
      }) as typeof fetch;

      const response = await worker.fetch(
        new Request("https://proxy.example.com/hf-transformers/onnx-community/moonshine-base-ONNX/resolve/main/config.json"),
        ENV,
      );

      expect(response.status).toBe(200);
      expect(response.headers.get("X-Proxy-Route")).toBe("hf-transformers");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("rejects repos that are not allowlisted", async () => {
    const response = await worker.fetch(
      new Request("https://proxy.example.com/hf-transformers/evil/repo/resolve/main/config.json"),
      ENV,
    );
    expect(response.status).toBe(403);
  });

  it("rejects non-wasm and non-allowlisted raw GitHub paths", async () => {
    const bad = await worker.fetch(
      new Request("https://proxy.example.com/gh-raw/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/notes.txt"),
      ENV,
    );
    expect(bad.status).toBe(403);
    const wrongRepo = await worker.fetch(
      new Request("https://proxy.example.com/gh-raw/someone/else/main/web-llm-models/x.wasm"),
      ENV,
    );
    expect(wrongRepo.status).toBe(403);
  });
});
