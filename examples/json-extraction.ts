/**
 * Example 2 — Schema-constrained JSON extraction.
 *
 * `buildExtractionMessages` assembles the system + user turns for extraction, embedding the schema.
 * Passing the same schema to `generateText({ schema })` makes WebLLM decode against a grammar; for
 * prompt-only backends the schema still guides via the prompt. `parseJsonFromModel` tolerates the
 * usual model quirks (code fences, trailing prose) and `validateAgainstJsonSchema` returns a list of
 * problems ([] means valid).
 */
import {
  BrowserAI,
  buildExtractionMessages,
  parseJsonFromModel,
  prettyJson,
  validateAgainstJsonSchema,
  WebGPUUnavailableError,
} from "@missionsquad/browserai";
import { byId, fillModelSelect, log, setProgress } from "./shared.js";

const modelSelect = byId<HTMLSelectElement>("model");
const loadBtn = byId<HTMLButtonElement>("load");
const extractBtn = byId<HTMLButtonElement>("extract");
const progress = byId<HTMLProgressElement>("progress");
const progressLabel = byId("progressLabel");
const logBox = byId("log");
const textInput = byId<HTMLTextAreaElement>("text");
const schemaInput = byId<HTMLTextAreaElement>("schema");
const rawOut = byId("raw");
const parsedOut = byId("parsed");
const validationOut = byId("validation");

// A compact JSON Schema for the demo. Any valid JSON Schema (draft-07 style) works.
const DEMO_SCHEMA = {
  type: "object",
  properties: {
    name: { type: "string" },
    nationality: { type: ["string", "null"] },
    knownFor: { type: "string" },
    skills: { type: "array", items: { type: "string" } },
  },
  required: ["name", "knownFor", "skills"],
  additionalProperties: false,
};
schemaInput.value = JSON.stringify(DEMO_SCHEMA, null, 2);

const ai = new BrowserAI();
ai.on("loadprogress", ({ progress: p, status }) => setProgress(progress, progressLabel, p, status));
// Skip the per-chunk "initiate/download/progress/done <file>" statuses Transformers.js models emit.
ai.on("status", ({ message }) => {
  if (!/^(initiate|download|progress|done) /.test(message)) log(logBox, message);
});

fillModelSelect(modelSelect, "text", "Qwen3.5-0.8B-q4f16_1-MLC");

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  extractBtn.disabled = true;
  try {
    log(logBox, `Loading ${modelSelect.value}…`);
    await ai.load(modelSelect.value);
    log(logBox, "Ready.");
    extractBtn.disabled = false;
  } catch (error) {
    log(logBox, error instanceof WebGPUUnavailableError ? "WebGPU is required for this model." : `Load failed: ${(error as Error).message}`);
  } finally {
    loadBtn.disabled = false;
  }
});

extractBtn.addEventListener("click", async () => {
  let schema: unknown;
  try {
    schema = JSON.parse(schemaInput.value);
  } catch {
    validationOut.textContent = "The schema box does not contain valid JSON.";
    return;
  }

  extractBtn.disabled = true;
  rawOut.textContent = "";
  parsedOut.textContent = "";
  validationOut.textContent = "extracting…";
  try {
    const messages = buildExtractionMessages(textInput.value, schema);
    const { text } = await ai.generateText(messages, {
      schema,
      onDelta: (full) => {
        rawOut.textContent = full;
      },
    });
    rawOut.textContent = text;

    const parsed = parseJsonFromModel(text);
    if (!parsed.ok) {
      parsedOut.textContent = `Could not parse JSON: ${parsed.error}`;
      validationOut.textContent = "";
      return;
    }
    parsedOut.textContent = prettyJson(parsed.value);
    const problems = validateAgainstJsonSchema(parsed.value, schema);
    validationOut.className = problems.length === 0 ? "stat ok" : "stat error";
    validationOut.textContent = problems.length === 0 ? "✓ valid against the schema" : `✗ ${problems.length} problem(s): ${problems.join("; ")}`;
  } catch (error) {
    validationOut.textContent = `Error: ${(error as Error).message}`;
  } finally {
    extractBtn.disabled = false;
  }
});
