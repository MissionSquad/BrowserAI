import { describe, expect, it } from "vitest";
import { detectLikelyRepetition, extractJsonCandidate, parseJsonFromModel } from "../src/json.js";
import { validateAgainstJsonSchema } from "../src/schema-validator.js";
import { DEFAULT_EXTRACTION_SCHEMA } from "../src/schema.js";
import { stripThinkTags } from "../src/text-utils.js";

describe("json helpers", () => {
  it("extracts fenced JSON", () => {
    expect(extractJsonCandidate('```json\n{"ok":true}\n```')).toBe('{"ok":true}');
  });

  it("extracts JSON after thinking tags", () => {
    expect(extractJsonCandidate('<think>hidden</think>\n{"ok":true}')).toBe('{"ok":true}');
  });

  it("extracts the first balanced JSON object when stray text follows", () => {
    expect(extractJsonCandidate('{"ok":true}\nextra')).toBe('{"ok":true}');
  });

  it("parses model JSON", () => {
    const result = parseJsonFromModel('{"summary":"done"}');
    expect(result.ok).toBe(true);
  });

  it("parses the first complete object with a warning if a model repeats after it", () => {
    const result = parseJsonFromModel('{"ok":true}\n{"ok":true}');
    expect(result.ok).toBe(true);
    expect(result.warnings.length).toBeGreaterThan(0);
  });

  it("validates against the default extraction schema", () => {
    const errors = validateAgainstJsonSchema(
      {
        summary: "A customer needs help.",
        language: "English",
        sentiment: "neutral",
        entities: [],
        intents: [],
        action_items: [],
        risk_flags: [],
        confidence: 0.8,
      },
      DEFAULT_EXTRACTION_SCHEMA,
    );

    expect(errors).toEqual([]);
  });

  it("reports schema validation errors", () => {
    const errors = validateAgainstJsonSchema({ summary: "missing most fields" }, DEFAULT_EXTRACTION_SCHEMA);
    expect(errors.some((error) => error.includes("$.language is required"))).toBe(true);
  });

  it("detects obvious repetition", () => {
    const warnings = detectLikelyRepetition(
      "same repeated phrase same repeated phrase same repeated phrase same repeated phrase",
    );
    expect(warnings.length).toBeGreaterThan(0);
  });
});

describe("stripThinkTags", () => {
  it("removes closed think blocks", () => {
    expect(stripThinkTags("<think>reasoning</think>Hello there.")).toBe("Hello there.");
  });

  it("truncates at an unclosed think block while streaming", () => {
    expect(stripThinkTags("Hello.<think>still reason")).toBe("Hello.");
  });

  it("leaves plain text alone", () => {
    expect(stripThinkTags("Just an answer.")).toBe("Just an answer.");
  });
});
