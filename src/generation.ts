import type { ResponseFormat } from "@mlc-ai/web-llm";
import type { RuntimePreset } from "./models.js";
import { stripSchemaMetadata } from "./schema.js";

export type JsonMode = "schema" | "json_object" | "none";

export type RuntimeParameters = RuntimePreset & {
  jsonMode: JsonMode;
  latencyBreakdown: boolean;
};

export function runtimeParamsFromPreset(preset: RuntimePreset): RuntimeParameters {
  return {
    ...preset,
    jsonMode: "schema",
    latencyBreakdown: true,
  };
}

export function buildResponseFormat(mode: JsonMode, schema: unknown): ResponseFormat | undefined {
  if (mode === "none") return undefined;
  if (mode === "json_object") return { type: "json_object" };

  return {
    type: "json_object",
    schema: JSON.stringify(stripSchemaMetadata(schema)),
  };
}

export function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export function parseOptionalInteger(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}
