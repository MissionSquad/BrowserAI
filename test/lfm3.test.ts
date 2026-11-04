import { describe, expect, it, vi } from "vitest";
import { describeImage, loadMultimodal, type MultimodalTransformers } from "../src/multimodal.js";
import { generate, type EngineHandles, type TransformersTokenizer } from "../src/inference.js";
import { runtimeParamsFromPreset } from "../src/generation.js";
import { getModelPreset, modelSlots } from "../src/models.js";

const variants = [
  { id: "onnx-community/LFM3-1.7B-ONNX", original: "https://huggingface.co/LiquidAI/LFM3-1.7B" },
];

describe.each(variants)("LFM3 text ONNX: $id", ({ id, original }) => {
  const preset = getModelPreset(id);

  it("uses the published q4f16 artifact as a text-only model", () => {
    expect(preset.id).toBe(id);
    expect(preset.backend).toBe("transformers-js");
    expect(preset.transformersDtype).toBe("q4f16");
    expect(preset.officialRepo).toBe(original);
    expect(preset.artifactRepo).toBe(`https://huggingface.co/${id}`);
    expect(modelSlots(preset)).toEqual(["text"]);
    expect(preset.defaultRuntime.disableThinking).toBe(true);
    expect(preset.defaultRuntime.repetitionPenalty).toBe(1);
  });

  it.each([true, false])("counts the actual chat prompt when disableThinking=%s", async (disableThinking) => {
    const applyTemplate = vi.fn((_messages: unknown, options?: Record<string, unknown>) =>
      options?.enable_thinking === false ? [1, 2, 3, 4, 5, 6] : [1, 2, 3],
    );
    const tokenizer: TransformersTokenizer = Object.assign(() => [42], {
      apply_chat_template: applyTemplate,
      encode: () => [42],
    });
    const generator = vi.fn(async () => [{ generated_text: "42" }]);
    const handles: EngineHandles = {
      engine: null,
      backend: "transformers-js",
      modelId: preset.id,
      transformersGenerator: generator,
      transformersTokenizer: tokenizer,
      transformersModule: null,
    };
    const messages = [{ role: "user" as const, content: "What is 17 + 25?" }];
    const result = await generate(handles, messages, {
      ...runtimeParamsFromPreset(preset.defaultRuntime), disableThinking,
    }, preset);

    expect(generator).toHaveBeenCalledWith(messages, expect.objectContaining({
      tokenizer_encode_kwargs: { enable_thinking: !disableThinking },
    }));
    expect(applyTemplate).toHaveBeenCalledWith(messages, {
      tokenize: true, add_generation_prompt: true, enable_thinking: !disableThinking,
    });
    expect(result.text).toBe("42");
    expect(result.stats.promptTokens).toBe(disableThinking ? 6 : 3);
    expect(result.stats.totalTokens).toBe(disableThinking ? 7 : 4);
    expect(result.stats.backend).toBe("transformers-js");
  });
});

const visionId = "onnx-community/LFM3-VL-2B-ONNX";
it("keeps both Liquid presets experimental with attributed sources and estimated memory", () => {
  for (const id of [variants[0].id, visionId]) {
    const preset = getModelPreset(id);
    expect(preset.id).toBe(id);
    expect(preset.brand).toBe("Liquid");
    expect(preset.stability).toBe("experimental");
    expect(preset.vramSource).toBe("estimated");
    expect(preset.artifactRepo).toBe(`https://huggingface.co/${id}`);
  }
  const vision = getModelPreset(visionId);
  expect(vision.officialRepo).toBe("https://huggingface.co/LiquidAI/LFM3-VL-2B");
  expect(modelSlots(vision)).toEqual(["vision"]);
  expect(vision.modalityIn).toEqual(["image", "text"]);
  expect(vision.modalityOut).toEqual(["text"]);
  expect(vision.mmRuntime).toBe("vlm-imagetext");
});

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

it("uses the LFM3-VL image-first processor without adding a second BOS", async () => {
  const f = fixtures();
  const preset = getModelPreset(visionId);
  const handle = await loadMultimodal(preset, f.transformers, "webgpu", () => {});
  expect(f.loader.from_pretrained).toHaveBeenCalledWith(visionId, expect.objectContaining({
    device: "webgpu", dtype: { vision_encoder: "fp16", embed_tokens: "fp16", decoder_model_merged: "q4f16" },
  }));
  // Avoid DOM-dependent resizing; preprocessing itself remains exercised with the actual handle.
  if (handle.kind !== "vlm") throw new Error("Expected VLM handle");
  handle.maxPixels = 0;
  await describeImage(handle, "source", "Describe this image");
  expect(f.processor.apply_chat_template).toHaveBeenCalledWith(expect.any(Array), {
    add_generation_prompt: true, enable_thinking: false,
  });
  expect(f.processor).toHaveBeenCalledWith("image", "prompt", { add_special_tokens: false });
  expect(f.model.generate).toHaveBeenCalledWith(expect.objectContaining({ pixel_values: "pixels", do_sample: true, temperature: 0.1, top_k: 50, repetition_penalty: 1.1 }));
});

it("keeps ordinary image-text processors text-first without a thinking override", async () => {
  const f = fixtures();
  const handle = await loadMultimodal(getModelPreset("onnx-community/Qwen3-VL-2B-Instruct-ONNX"), f.transformers, "webgpu", () => {});
  if (handle.kind !== "vlm") throw new Error("Expected VLM handle");
  handle.maxPixels = 0;
  await describeImage(handle, "source", "Describe");
  expect(f.processor).toHaveBeenCalledWith("prompt", "image");
  expect(f.model.generate).toHaveBeenCalledWith(expect.objectContaining({ do_sample: false }));
  expect(f.model.generate).not.toHaveBeenCalledWith(expect.objectContaining({ temperature: expect.any(Number) }));
  expect(f.processor.apply_chat_template).toHaveBeenCalledWith(expect.any(Array), { add_generation_prompt: true });
});
