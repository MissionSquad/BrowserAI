import { describe, expect, it, vi } from "vitest";
import { generate, type EngineHandles, type TransformersTokenizer } from "../src/inference.js";
import { runtimeParamsFromPreset } from "../src/generation.js";
import { getModelPreset, modelSlots } from "../src/models.js";

const variants = [
  { id: "RASMUS/MiniCPM5-2B-ONNX", original: "https://huggingface.co/openbmb/MiniCPM5-2B" },
  { id: "j4ys0n/MiniCPM5-2B-heretic-abliterated-ONNX", original: "https://huggingface.co/insraq/MiniCPM5-2B-heretic-abliterated" },
];

describe.each(variants)("MiniCPM5 2B ONNX: $id", ({ id, original }) => {
  const preset = getModelPreset(id);

  it("uses the browser-tested q4f16 artifact as a text-only model", () => {
    expect(preset.id).toBe(id);
    expect(preset.backend).toBe("transformers-js");
    expect(preset.transformersDtype).toBe("q4f16");
    expect(preset.officialRepo).toBe(original);
    expect(preset.artifactRepo).toBe(`https://huggingface.co/${id}`);
    expect(modelSlots(preset)).toEqual(["text"]);
    expect(preset.defaultRuntime.disableThinking).toBe(true);
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

it("keeps regular and Heretic cache identities and history labels distinct", () => {
  const [regular, heretic] = variants.map(({ id }) => getModelPreset(id));
  expect(regular.id).not.toBe(heretic.id);
  expect(regular.artifactRepo).not.toBe(heretic.artifactRepo);
  expect(regular.shortName).not.toBe(heretic.shortName);
});
