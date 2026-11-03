import { describe, expect, it, vi } from "vitest";
import { classifyImage, describeImage, disposeMultimodal, loadMultimodal, type MultimodalTransformers } from "../src/multimodal.js";
import { getModelPreset } from "../src/models.js";

const docling = getModelPreset("onnx-community/granite-docling-258M-ONNX");
const nsfw = getModelPreset("onnx-community/nsfw_image_detection-ONNX");
function fixtures() {
  const processor = Object.assign(vi.fn(async () => ({ pixel_values: "pixels" })), {
    tokenizer: {}, apply_chat_template: vi.fn(() => "prompt"),
  });
  const streamerOptions: Array<Record<string, unknown>> = [];
  class Streamer { constructor(_tokenizer: unknown, options: Record<string, unknown> = {}) { streamerOptions.push(options); } }
  const model = { generate: vi.fn(async () => {
    const opts = streamerOptions.at(-1)!;
    (opts.callback_function as (text: string) => void)("<doctag><text>Hello</text></doctag><|end_of_text|>");
    (opts.token_callback_function as (tokens: number[]) => void)([1, 2]);
  }), dispose: vi.fn(async () => {}) };
  const classifier = Object.assign(vi.fn(async () => [{ label: "normal", score: 0.9 }, { label: "nsfw", score: 0.1 }]), { dispose: vi.fn(async () => {}) });
  const loader = { from_pretrained: vi.fn(async () => model) };
  const transformers = {
    pipeline: vi.fn(async () => classifier),
    AutoProcessor: { from_pretrained: vi.fn(async () => processor) },
    AutoModelForVision2Seq: loader, AutoModelForImageTextToText: loader,
    AutoModelForAudioTextToText: loader, Gemma4ForConditionalGeneration: loader,
    TextStreamer: Streamer, load_image: vi.fn(async () => "image"), read_audio: vi.fn(async () => new Float32Array()),
  } satisfies MultimodalTransformers;
  return { processor, model, classifier, loader, transformers, streamerOptions };
}
describe("document conversion and image classification", () => {
  it("loads Docling through vision2seq and preserves DocTags with its own generation settings", async () => {
    const f = fixtures();
    const handle = await loadMultimodal(docling, f.transformers, "webgpu", () => {});
    expect(f.loader.from_pretrained).toHaveBeenCalledWith(docling.id, expect.objectContaining({ dtype: "fp32", device: "webgpu" }));
    const delta = vi.fn();
    const result = await describeImage(handle, "source", "", delta);
    expect(f.processor.apply_chat_template).toHaveBeenCalledWith(expect.arrayContaining([expect.objectContaining({ content: [{ type: "image" }, { type: "text", text: "Convert this page to docling." }] })]), expect.anything());
    expect(f.processor).toHaveBeenCalledWith("prompt", ["image"], { do_image_splitting: true });
    expect(f.model.generate).toHaveBeenCalledWith(expect.objectContaining({ max_new_tokens: 4096 }));
    expect(f.streamerOptions[0].skip_special_tokens).toBe(false);
    expect(result.text).toBe("<doctag><text>Hello</text></doctag>");
    expect(delta).toHaveBeenCalledWith(result.text);
    expect(result.completionTokens).toBe(2);
  });
  it("keeps other vision models' special-token stripping and processor options", async () => {
    const f = fixtures();
    const handle = await loadMultimodal(getModelPreset("HuggingFaceTB/SmolVLM-256M-Instruct"), f.transformers, "webgpu", () => {});
    expect(f.loader.from_pretrained).toHaveBeenCalledWith("HuggingFaceTB/SmolVLM-256M-Instruct", expect.objectContaining({ dtype: { embed_tokens: "fp16", vision_encoder: "q4", decoder_model_merged: "q4" } }));
    await describeImage(handle, "source", "Describe");
    expect(f.processor).toHaveBeenCalledWith("prompt", ["image"], {});
    expect(f.streamerOptions[0].skip_special_tokens).toBe(true);
  });
  it("returns classifier labels with timing and disposes its pipeline", async () => {
    const f = fixtures();
    const handle = await loadMultimodal(nsfw, f.transformers, "wasm", () => {});
    expect(f.transformers.pipeline).toHaveBeenCalledWith("image-classification", nsfw.id, expect.objectContaining({ dtype: "fp32", device: "wasm" }));
    const result = await classifyImage(handle, "source");
    expect(f.classifier).toHaveBeenCalledWith("source", { top_k: null });
    expect(result.scores).toEqual([{ label: "normal", score: 0.9 }, { label: "nsfw", score: 0.1 }]);
    expect(result.endedMs).toBeGreaterThanOrEqual(result.startedMs);
    await expect(describeImage(handle, "source", "Describe")).rejects.toThrow("use classifyImage()");
    await disposeMultimodal(handle);
    expect(f.classifier.dispose).toHaveBeenCalledOnce();
  });
  it("cleans processor resources when the vision model load fails", async () => {
    const f = fixtures();
    const dispose = vi.fn(async () => {});
    Object.assign(f.processor, { dispose });
    f.loader.from_pretrained.mockRejectedValueOnce(new Error("model failed"));
    await expect(loadMultimodal(docling, f.transformers, "webgpu", () => {})).rejects.toThrow("model failed");
    expect(dispose).toHaveBeenCalledOnce();
  });
  it("rejects classifier dispatch on a generative model and malformed scores", async () => {
    const f = fixtures();
    const vlm = await loadMultimodal(docling, f.transformers, "webgpu", () => {});
    await expect(classifyImage(vlm, "source")).rejects.toThrow("cannot classify images");
    await expect(classifyImage({ kind: "image-classification", classifier: async () => [{ label: "nsfw", score: NaN }] }, "source")).rejects.toThrow("invalid label scores");
  });
});
