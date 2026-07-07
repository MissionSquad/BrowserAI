import { describe, expect, it } from "vitest";
import type { ModelTask } from "../src/models.js";
import { EMPTY_RESPONSE } from "../src/inference.js";
import {
  capPipelineHistory,
  missingPipelineModels,
  PIPELINE_HISTORY_MAX_TURNS,
  shouldSpeakReply,
  streamedDelta,
} from "../src/voice/helpers.js";

describe("missingPipelineModels", () => {
  it("reports all three chain slots when nothing is loaded", () => {
    expect(missingPipelineModels(new Set())).toEqual(["stt", "text", "tts"]);
  });

  it("reports only the unfilled slots, in chain order", () => {
    expect(missingPipelineModels(new Set<ModelTask>(["text"]))).toEqual(["stt", "tts"]);
    expect(missingPipelineModels(new Set<ModelTask>(["stt", "tts"]))).toEqual(["text"]);
  });

  it("reports nothing when the chain is complete (extra slots ignored)", () => {
    expect(missingPipelineModels(new Set<ModelTask>(["stt", "text", "tts", "vision"]))).toEqual([]);
  });
});

describe("shouldSpeakReply", () => {
  it("speaks normal replies", () => {
    expect(shouldSpeakReply("Hello there.")).toBe(true);
  });

  it("skips blank output and the empty-response sentinel", () => {
    expect(shouldSpeakReply("")).toBe(false);
    expect(shouldSpeakReply("   \n")).toBe(false);
    expect(shouldSpeakReply(EMPTY_RESPONSE)).toBe(false);
    expect(shouldSpeakReply(`  ${EMPTY_RESPONSE}  `)).toBe(false);
  });
});

describe("capPipelineHistory", () => {
  const entries = (count: number) => Array.from({ length: count }, (_, index) => index);

  it("returns short histories unchanged (same reference)", () => {
    const history = entries(4);
    expect(capPipelineHistory(history)).toBe(history);
  });

  it("keeps only the most recent maxTurns exchanges", () => {
    const history = entries(PIPELINE_HISTORY_MAX_TURNS * 2 + 6);
    const capped = capPipelineHistory(history);
    expect(capped).toHaveLength(PIPELINE_HISTORY_MAX_TURNS * 2);
    expect(capped[capped.length - 1]).toBe(history[history.length - 1]);
    expect(capped[0]).toBe(history[6]);
  });

  it("honors a custom turn cap", () => {
    expect(capPipelineHistory(entries(10), 2)).toHaveLength(4);
  });
});

describe("streamedDelta", () => {
  it("returns the new suffix of a monotonic stream", () => {
    expect(streamedDelta("", "Hello")).toBe("Hello");
    expect(streamedDelta("Hello", "Hello there")).toBe(" there");
    expect(streamedDelta("Hello there", "Hello there")).toBe("");
  });

  it("returns empty when the text shrank (think-tag strip)", () => {
    expect(streamedDelta("Hello <think>rea", "Hello ")).toBe("");
    expect(streamedDelta("abc", "xyz")).toBe("");
  });
});
