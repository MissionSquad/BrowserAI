import { describe, expect, it } from "vitest";
import * as webllm from "@mlc-ai/web-llm";
import {
  CUSTOM_MODEL_RECORD_IDS,
  CUSTOM_MODEL_RECORDS,
  getModelPreset,
  MODEL_PRESETS,
  MODEL_RECORD_OVERRIDES,
  modelSlots,
} from "../src/models.js";
import { vendorForModel } from "../src/history.js";
import { PIPELINE_RECOMMENDED, PIPELINE_TASKS } from "../src/voice/helpers.js";

describe("modelSlots", () => {
  it("maps a plain text LLM to the Text slot only", () => {
    expect(modelSlots(getModelPreset("Qwen3.5-0.8B-q4f16_1-MLC"))).toEqual(["text"]);
  });

  it("gives a VLM only the Image slot (VLMs can't run text-only via Transformers.js)", () => {
    expect(modelSlots(getModelPreset("HuggingFaceTB/SmolVLM-256M-Instruct"))).toEqual(["vision"]);
  });

  it("maps a Gemma multimodal model to Text + Image + Transcription", () => {
    // Order follows SLOT_ORDER: text, vision, stt, tts.
    expect(modelSlots(getModelPreset("onnx-community/gemma-4-E2B-it-ONNX"))).toEqual(["text", "vision", "stt"]);
  });

  it("maps an STT model to the Transcription slot only", () => {
    expect(modelSlots(getModelPreset("onnx-community/whisper-base"))).toEqual(["stt"]);
  });

  it("maps a TTS model to the Speech slot only", () => {
    expect(modelSlots(getModelPreset("onnx-community/Kokoro-82M-v1.0-ONNX"))).toEqual(["tts"]);
  });

  it("classifies re-hosted artifacts under their true vendor", () => {
    expect(vendorForModel(getModelPreset("onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX"))).toBe("mistral");
    expect(vendorForModel(getModelPreset("onnx-community/granite-speech-4.1-2b-ONNX"))).toBe("ibm");
    expect(vendorForModel(getModelPreset("onnx-community/parakeet-ctc-0.6b-ONNX"))).toBe("nvidia");
  });
});

describe("MODEL_PRESETS invariants", () => {
  it("has unique preset ids", () => {
    const ids = MODEL_PRESETS.map((preset) => preset.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("gives every Transformers.js preset an owner/repo-shaped id (the proxy allowlist derives from it)", () => {
    // The worker template allowlists /hf-transformers/{owner}/{repo} by splitting the preset id; an
    // id without an owner would silently be un-proxied and 403 on proxied deployments.
    for (const preset of MODEL_PRESETS.filter((entry) => entry.backend === "transformers-js")) {
      expect(preset.id.split("/").length, `${preset.id} must be owner/repo`).toBeGreaterThanOrEqual(2);
    }
  });

  it("only includes official prebuilt WebLLM records or explicit custom MLC records", () => {
    const prebuiltIds = new Set(webllm.prebuiltAppConfig.model_list.map((record) => record.model_id));
    const allowedIds = new Set([...prebuiltIds, ...CUSTOM_MODEL_RECORD_IDS]);
    expect(
      MODEL_PRESETS.filter((preset) => preset.backend === "webllm")
        .map((preset) => preset.id)
        .filter((id) => !allowedIds.has(id)),
    ).toEqual([]);
  });

  it("defines the custom Gemma records with WebGPU wasm libraries and fixed-context overrides", () => {
    expect(CUSTOM_MODEL_RECORDS.length).toBe(2);
    for (const record of CUSTOM_MODEL_RECORDS) {
      expect(record.model).toContain("https://huggingface.co/welcoma/");
      expect(record.model_lib).toMatch(/-webgpu\.wasm$/);
      expect(record.required_features).toContain("shader-f16");
      expect(record.overrides).toMatchObject({ context_window_size: 4096, sliding_window_size: -1 });
    }
    expect(MODEL_RECORD_OVERRIDES["gemma3-1b-it-q4f16_1-MLC"]).toMatchObject({
      context_window_size: 4096,
      sliding_window_size: -1,
    });
  });
});

describe("voice pipeline recommended models", () => {
  it("references real catalog presets", () => {
    for (const task of PIPELINE_TASKS) {
      const preset = MODEL_PRESETS.find((entry) => entry.id === PIPELINE_RECOMMENDED[task]);
      expect(preset, `${PIPELINE_RECOMMENDED[task]} must exist in the catalog`).toBeTruthy();
    }
  });

  it("occupies disjoint capability slots so loading all three never evicts another", () => {
    const seen = new Set<string>();
    for (const task of PIPELINE_TASKS) {
      const preset = MODEL_PRESETS.find((entry) => entry.id === PIPELINE_RECOMMENDED[task])!;
      for (const slot of modelSlots(preset)) {
        expect(seen.has(slot), `slot ${slot} claimed twice`).toBe(false);
        seen.add(slot);
      }
    }
  });
});
