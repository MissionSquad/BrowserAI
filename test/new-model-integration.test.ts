import { afterEach, describe, expect, it, vi } from "vitest";
import { getModelPreset, modelSlots, supportsWasmFallback } from "../src/models.js";
import worker from "../worker-template/cloudflare-worker";
import { RunHistoryStore } from "../src/history.js";

import { transcribe, disposeMultimodal } from "../src/multimodal";
import { getCachedModelStatus, deleteTransformersModelFromBrowserCache } from "../src/model-cache";

const speech = [
  ["enriqueag/phonon-2.5-onnx", ["config.json", "vocab.txt", "encoder.exact2x2.onnx", "decoder_joint.int8.onnx"]],
  ["istupakov/parakeet-tdt-0.6b-v3-onnx", ["config.json", "vocab.txt", "nemo128.onnx", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]],
] as const;
const env = { ASSETS: { fetch: async () => new Response("asset") } };
afterEach(() => vi.unstubAllGlobals());

describe("new model integration contracts", () => {
  it.each([
    [speech[0][0], "stt", "stt-tdt", "onnx-runtime"],
    [speech[1][0], "stt", "stt-tdt", "onnx-runtime"],
    ["onnx-community/granite-docling-258M-ONNX", "vision", "vlm-vision2seq", "transformers-js"],
    ["onnx-community/nsfw_image_detection-ONNX", "vision", "image-classification", "transformers-js"],
  ])("routes %s to its capability slot and runtime", (id, slot, runtime, backend) => {
    const preset = getModelPreset(id);
    expect(modelSlots(preset)).toEqual([slot]);
    expect(preset.mmRuntime).toBe(runtime);
    expect(preset.backend).toBe(backend);
    expect(preset.artifactRepo).toBe(`https://huggingface.co/${id}`);
  });

  it.each([
    ["istupakov/parakeet-tdt-0.6b-v3-onnx", true],
    ["enriqueag/phonon-2.5-onnx", false],
    ["onnx-community/nsfw_image_detection-ONNX", true],
    ["onnx-community/granite-docling-258M-ONNX", false],
  ] as const)("advertises the usable WASM fallback for %s", (id, supported) => {
    expect(supportsWasmFallback(getModelPreset(id))).toBe(supported);
  });

  it("publishes both speech repositories in proxy discovery", async () => {
    const response = await worker.fetch(new Request("https://example.com/__worker-health"), env);
    const health = await response.json() as { allowedHuggingFaceRepos: string[] };
    expect(health.allowedHuggingFaceRepos).toEqual(expect.arrayContaining(speech.map(([repo]) => repo)));
  });

  for (const [repo, files] of speech) {
    it.each(files)(`preserves the exact ${repo}/%s upstream path and metadata`, async (file) => {
      const path = `${repo}/resolve/main/${file}`;
      const upstream = vi.fn(async () => new Response(null, { headers: { "Content-Type": "application/octet-stream", "Content-Length": "12345", ETag: '"model-v1"' } }));
      vi.stubGlobal("fetch", upstream);
      const response = await worker.fetch(new Request(`https://example.com/hf-transformers/${path}`, { method: "HEAD" }), env);
      expect(response.status).toBe(200);
      expect(response.headers.get("X-Upstream-URL")).toBe(`https://huggingface.co/${path}`);
      expect(response.headers.get("Content-Length")).toBe("12345");
      expect(response.headers.get("ETag")).toBe('"model-v1"');
      expect(upstream).toHaveBeenCalledOnce();
    });
  }

  it("denies lookalike repositories without contacting upstream", async () => {
    const upstream = vi.fn(); vi.stubGlobal("fetch", upstream);
    const response = await worker.fetch(new Request("https://example.com/hf-transformers/enriqueag/phonon-2.5-onnx-untrusted/resolve/main/config.json"), env);
    expect(response.status).toBe(403); expect(upstream).not.toHaveBeenCalled();
  });

  it("dispatches speech PCM, incremental output, cancellation and disposal to the TDT handle", async () => {
    const pcm = Float32Array.of(0.1, 0.2); const delta = vi.fn(); const signal = new AbortController().signal;
    const result = { text: "hello", segments: [{ start: 0, end: 1, text: "hello" }] };
    const model = { transcribe: vi.fn(async () => result), dispose: vi.fn(async () => {}) };
    const handle = { kind: "tdt" as const, model };
    expect(await transcribe(handle, pcm, delta, signal)).toBe(result);
    expect(model.transcribe).toHaveBeenCalledWith(pcm, delta, signal);
    await disposeMultimodal(handle); expect(model.dispose).toHaveBeenCalledOnce();
    const aborted = new AbortController(); aborted.abort();
    await expect(transcribe(handle, pcm, delta, aborted.signal)).rejects.toThrow();
    expect(model.transcribe).toHaveBeenCalledOnce();
  });

  it("finds and removes TDT artifacts from the shared model cache without touching other models", async () => {
    const repo = speech[0][0];
    const own = new Request(`https://example.com/hf-transformers/${repo}/resolve/main/encoder.exact2x2.onnx`);
    const other = new Request("https://huggingface.co/another/model/resolve/main/model.onnx");
    const keys = [own, other];
    const cache = { keys: vi.fn(async () => keys), delete: vi.fn(async () => true) };
    vi.stubGlobal("caches", { open: vi.fn(async (scope: string) => scope === "transformers-cache" ? cache : { keys: async () => [] }) });
    const status = await getCachedModelStatus({ modelId: repo, repoLabels: [repo], artifactUrls: [] });
    expect(status.downloaded).toBe(true); expect(status.matchedEntries).toBe(1);
    expect(await deleteTransformersModelFromBrowserCache(repo)).toBe(1);
    expect(cache.delete).toHaveBeenCalledExactlyOnceWith(own);
  });

  it("keeps ONNX Runtime attribution after persisting and reloading history", () => {
    const data = new Map<string, string>();
    const storage = { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => { data.set(key, value); }, removeItem: (key: string) => { data.delete(key); } };
    const record = { ts: 1, model: "Phonon 2.5", vendor: "other" as const, backend: "onnx-runtime" as const, case: "voice" as const, promptTokens: 0, completionTokens: 2, decode: 2, ttft: 0, totalMs: 1000, measured: 2 };
    const store = new RunHistoryStore({ storage }); store.append(record);
    expect(new RunHistoryStore({ storage }).records[0]).toEqual(record);
  });
});
