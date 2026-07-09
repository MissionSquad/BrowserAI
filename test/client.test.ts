import { beforeEach, describe, expect, it, vi } from "vitest";

/* ------------------------------------------------------------------ */
/* Module mocks: fake WebGPU probe + fake engines                      */
/* ------------------------------------------------------------------ */

const hardwareState = { webgpuSupported: true, chromiumBased: true };

vi.mock("../src/hardware.js", () => ({
  isChromiumBased: () => hardwareState.chromiumBased,
  collectHardwareSnapshot: async () => ({
    webgpuSupported: hardwareState.webgpuSupported,
    webgpuReason: hardwareState.webgpuSupported ? "ok" : "no adapter",
    features: [],
    limits: {},
    secureContext: true,
    crossOriginIsolated: false,
    userAgent: "vitest",
  }),
}));

type FakeEngine = {
  chat: { completions: { create: ReturnType<typeof vi.fn> } };
  resetChat: ReturnType<typeof vi.fn>;
  unload: ReturnType<typeof vi.fn>;
  runtimeStatsText: ReturnType<typeof vi.fn>;
};

const created: FakeEngine[] = [];
let activeGenerations = 0;
let maxConcurrentGenerations = 0;

function makeFakeEngine(): FakeEngine {
  const engine: FakeEngine = {
    chat: {
      completions: {
        create: vi.fn(async () => {
          activeGenerations += 1;
          maxConcurrentGenerations = Math.max(maxConcurrentGenerations, activeGenerations);
          await new Promise((resolve) => setTimeout(resolve, 5));
          activeGenerations -= 1;
          return {
            choices: [{ message: { content: "hello from the fake engine" }, finish_reason: "stop" }],
            usage: { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 },
          };
        }),
      },
    },
    resetChat: vi.fn(async () => undefined),
    unload: vi.fn(async () => undefined),
    runtimeStatsText: vi.fn(async () => ""),
  };
  created.push(engine);
  return engine;
}

vi.mock("@mlc-ai/web-llm", () => ({
  prebuiltAppConfig: {
    model_list: [
      {
        model: "https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC",
        model_id: "Qwen3.5-0.8B-q4f16_1-MLC",
        model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/qwen.wasm",
      },
      {
        model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC",
        model_id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
        model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/llama.wasm",
      },
    ],
  },
  CreateMLCEngine: vi.fn(async (_modelId: string, engineConfig?: { initProgressCallback?: (r: { progress: number; text: string }) => void }) => {
    engineConfig?.initProgressCallback?.({ progress: 0.5, text: "halfway" });
    return makeFakeEngine();
  }),
  CreateWebWorkerMLCEngine: vi.fn(async () => makeFakeEngine()),
}));

const transformersCalls: Array<{ task: string; model: string; options: Record<string, unknown> }> = [];

vi.mock("@huggingface/transformers", () => ({
  env: {},
  pipeline: vi.fn(async (task: string, model: string, options: Record<string, unknown>) => {
    transformersCalls.push({ task, model, options });
    const fake = async () => [{ generated_text: "fake" }];
    return fake;
  }),
  TextStreamer: class {},
  AutoProcessor: { from_pretrained: vi.fn(async () => ({})) },
  AutoModelForVision2Seq: { from_pretrained: vi.fn(async () => ({})) },
  AutoModelForImageTextToText: { from_pretrained: vi.fn(async () => ({})) },
  AutoModelForAudioTextToText: { from_pretrained: vi.fn(async () => ({})) },
  Gemma4ForConditionalGeneration: { from_pretrained: vi.fn(async () => ({})) },
  load_image: vi.fn(),
  read_audio: vi.fn(),
}));

import { BrowserAI } from "../src/client.js";
import { UnknownModelError, WebGPUUnavailableError } from "../src/errors.js";

beforeEach(() => {
  hardwareState.webgpuSupported = true;
  hardwareState.chromiumBased = true;
  created.length = 0;
  transformersCalls.length = 0;
  activeGenerations = 0;
  maxConcurrentGenerations = 0;
});

describe("BrowserAI slot manager", () => {
  it("loads a WebLLM model into the Text slot and reports occupancy", async () => {
    const ai = new BrowserAI();
    const progressEvents: number[] = [];
    ai.on("loadprogress", ({ progress }) => progressEvents.push(progress));

    const model = await ai.load("Qwen3.5-0.8B-q4f16_1-MLC");
    expect(model.backend).toBe("webllm");
    expect(model.slots).toEqual(["text"]);
    expect(ai.textModel()?.modelId).toBe("Qwen3.5-0.8B-q4f16_1-MLC");
    expect([...ai.occupiedSlots()]).toEqual(["text"]);
    expect(progressEvents).toContain(1);
  });

  it("throws UnknownModelError for an id not in the catalog (never loads a fallback)", async () => {
    const ai = new BrowserAI();
    await expect(ai.load("not-a-model")).rejects.toBeInstanceOf(UnknownModelError);
    expect(ai.loadedModels).toHaveLength(0);
  });

  it("evicts the conflicting slot owner when loading a same-slot model", async () => {
    const ai = new BrowserAI();
    await ai.load("Qwen3.5-0.8B-q4f16_1-MLC");
    const firstEngine = created[0];
    const unloaded: string[] = [];
    ai.on("modelunloaded", ({ modelId }) => unloaded.push(modelId));

    await ai.load("Llama-3.2-1B-Instruct-q4f16_1-MLC");
    expect(firstEngine.unload).toHaveBeenCalled();
    expect(unloaded).toEqual(["Qwen3.5-0.8B-q4f16_1-MLC"]);
    expect(ai.loadedModels).toHaveLength(1);
    expect(ai.textModel()?.modelId).toBe("Llama-3.2-1B-Instruct-q4f16_1-MLC");
  });

  it("keeps disjoint-slot models resident together and computes isLoadable", async () => {
    const ai = new BrowserAI();
    await ai.load("Qwen3.5-0.8B-q4f16_1-MLC");
    await ai.load("onnx-community/whisper-base");
    expect(ai.loadedModels).toHaveLength(2);
    expect([...ai.occupiedSlots()].sort()).toEqual(["stt", "text"]);
    expect(ai.isLoadable(ai.getPreset("onnx-community/Kokoro-82M-v1.0-ONNX")!)).toBe(true);
    expect(ai.isLoadable(ai.getPreset("Llama-3.2-1B-Instruct-q4f16_1-MLC")!)).toBe(false);
  });

  it("unload tears the engine down and frees the slot", async () => {
    const ai = new BrowserAI();
    await ai.load("Qwen3.5-0.8B-q4f16_1-MLC");
    await ai.unload("Qwen3.5-0.8B-q4f16_1-MLC");
    expect(created[0].unload).toHaveBeenCalled();
    expect(ai.loadedModels).toHaveLength(0);
    await expect(ai.unload("Qwen3.5-0.8B-q4f16_1-MLC")).resolves.toBeUndefined(); // no-op
  });

  it("gates non-multimodal loads on WebGPU but lets ASR/TTS presets fall back to wasm", async () => {
    hardwareState.webgpuSupported = false;
    const ai = new BrowserAI();
    await expect(ai.load("Qwen3.5-0.8B-q4f16_1-MLC")).rejects.toBeInstanceOf(WebGPUUnavailableError);

    await ai.load("onnx-community/whisper-base");
    const call = transformersCalls.find((entry) => entry.task === "automatic-speech-recognition");
    expect(call?.options.device).toBe("wasm");
  });

  it("prefers wasm for ASR/TTS presets off-Chromium even when the WebGPU probe passes", async () => {
    hardwareState.chromiumBased = false;
    const ai = new BrowserAI();
    await ai.load("onnx-community/whisper-base");
    const call = transformersCalls.find((entry) => entry.task === "automatic-speech-recognition");
    expect(call?.options.device).toBe("wasm");
    // The wasm EP must cap graph optimization: ORT 1.26-dev's extended QDQ rewrites break the
    // quantized ASR decoders at session creation ("Missing required scale" / MatMulNBits).
    expect(call?.options.session_options).toEqual({ graphOptimizationLevel: "basic" });
  });

  it("does NOT offer a wasm fallback for VLM/Gemma/audio-LLM presets (their recipes hardcode WebGPU)", async () => {
    hardwareState.webgpuSupported = false;
    const ai = new BrowserAI();
    await expect(ai.load("HuggingFaceTB/SmolVLM-256M-Instruct")).rejects.toBeInstanceOf(WebGPUUnavailableError);
    await expect(ai.load("onnx-community/granite-speech-4.1-2b-ONNX")).rejects.toBeInstanceOf(WebGPUUnavailableError);
  });

  it("configures the Transformers.js environment for direct downloads by default", async () => {
    const ai = new BrowserAI();
    await ai.load("onnx-community/whisper-base");
    const transformers = await import("@huggingface/transformers");
    const env = (transformers as unknown as { env: Record<string, unknown> }).env;
    expect(env.remoteHost).toBe("https://huggingface.co/");
    expect(env.cacheKey).toBe("transformers-cache");
  });
});

describe("BrowserAI inference façade", () => {
  it("generates text with the loaded model's preset defaults", async () => {
    const ai = new BrowserAI();
    await ai.load("Qwen3.5-0.8B-q4f16_1-MLC");
    const result = await ai.generateText([{ role: "user", content: "hi" }], { runtime: { jsonMode: "none" } });
    expect(result.text).toBe("hello from the fake engine");
    expect(result.stats.backend).toBe("webllm");
    expect(created[0].resetChat).toHaveBeenCalled();
  });

  it("serializes concurrent runs against the same model", async () => {
    const ai = new BrowserAI();
    await ai.load("Qwen3.5-0.8B-q4f16_1-MLC");
    await Promise.all([
      ai.generateText([{ role: "user", content: "one" }], { runtime: { jsonMode: "none" } }),
      ai.generateText([{ role: "user", content: "two" }], { runtime: { jsonMode: "none" } }),
      ai.generateText([{ role: "user", content: "three" }], { runtime: { jsonMode: "none" } }),
    ]);
    expect(created[0].chat.completions.create).toHaveBeenCalledTimes(3);
    expect(maxConcurrentGenerations).toBe(1);
  });

  it("queues unload behind an in-flight run on the same model", async () => {
    const ai = new BrowserAI();
    await ai.load("Qwen3.5-0.8B-q4f16_1-MLC");
    const generation = ai.generateText([{ role: "user", content: "hi" }], { runtime: { jsonMode: "none" } });
    const unloadPromise = ai.unload("Qwen3.5-0.8B-q4f16_1-MLC");
    await expect(generation).resolves.toMatchObject({ text: "hello from the fake engine" });
    await unloadPromise;
    expect(ai.loadedModels).toHaveLength(0);
  });

  it("createStreamingTts acquires the TTS lock (queues behind synthesize; holds until release)", async () => {
    const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
    let releaseGenerate!: () => void;
    const generateGate = new Promise<void>((resolve) => {
      releaseGenerate = resolve;
    });
    let firstGenerate = true;
    const fakeRawAudio = { audio: new Float32Array(8), sampling_rate: 24000, toBlob: () => new Blob() };
    const fakeKokoro = {
      KokoroTTS: {
        from_pretrained: async () => ({
          generate: async () => {
            if (firstGenerate) {
              firstGenerate = false;
              await generateGate;
            }
            return fakeRawAudio;
          },
          stream: () => ({
            async *[Symbol.asyncIterator]() {
              // Ends immediately; the lock lifetime is governed by session.release(), not the stream.
            },
          }),
        }),
      },
      TextSplitterStream: class {
        push(): void {}
        flush(): void {}
        close(): void {}
      },
    };

    const ai = new BrowserAI({ kokoro: { load: async () => fakeKokoro as never } });
    await ai.load("onnx-community/Kokoro-82M-v1.0-ONNX");

    const firstSynthesis = ai.synthesize("one"); // occupies the Kokoro run chain (blocked on the gate)
    let sessionAcquired = false;
    const sessionPromise = ai.createStreamingTts().then((session) => {
      sessionAcquired = true;
      return session;
    });
    await tick();
    expect(sessionAcquired).toBe(false); // must wait for the in-flight run

    releaseGenerate();
    await firstSynthesis;
    const session = await sessionPromise;
    expect(sessionAcquired).toBe(true);

    let secondDone = false;
    const secondSynthesis = ai.synthesize("two").then((result) => {
      secondDone = true;
      return result;
    });
    await tick();
    expect(secondDone).toBe(false); // the session holds the lock until released

    session.release();
    await secondSynthesis;
    expect(secondDone).toBe(true);
  });

  it("throws MissingModelError-flavored errors when the needed slot is empty", async () => {
    const ai = new BrowserAI();
    await expect(ai.generateText([{ role: "user", content: "hi" }])).rejects.toThrow(/text/);
    await expect(ai.transcribe(new Float32Array(16000))).rejects.toThrow(/stt/);
    await expect(ai.synthesize("hello")).rejects.toThrow(/tts/);
    await expect(ai.describeImage("data:", "what is this?")).rejects.toThrow(/vision/);
  });
});
