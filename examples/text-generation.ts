/**
 * Example 1 — Text generation with streaming and metrics.
 *
 * Shows the smallest useful flow: construct a BrowserAI client, subscribe to load progress, load a
 * text model, then stream a chat completion. `generateText` resolves with the full text plus a
 * RuntimeStats record; `onDelta` receives the FULL accumulated text on every token so the UI can
 * render as it goes.
 */
import { BrowserAI, stripThinkTags, WebGPUUnavailableError } from "@missionsquad/browserai";
import type { ChatMessage } from "@missionsquad/browserai";
import { byId, fillModelSelect, formatStats, log, setProgress } from "./shared.js";

const modelSelect = byId<HTMLSelectElement>("model");
const loadBtn = byId<HTMLButtonElement>("load");
const generateBtn = byId<HTMLButtonElement>("generate");
const progress = byId<HTMLProgressElement>("progress");
const progressLabel = byId("progressLabel");
const logBox = byId("log");
const systemInput = byId<HTMLInputElement>("system");
const promptInput = byId<HTMLTextAreaElement>("prompt");
const output = byId("output");
const statsLine = byId("stats");

// One client owns the model catalog, the loaded models, and the four capability slots.
const ai = new BrowserAI();

// Two ways to observe loading: the "loadprogress" event (below) and the per-call onProgress option.
ai.on("loadprogress", ({ progress: p, status }) => setProgress(progress, progressLabel, p, status));
// Transformers.js models emit a per-chunk "initiate/download/progress/done <file>" status for every
// artifact — thousands of lines per load. Keep the meaningful messages; the bar above shows progress.
ai.on("status", ({ message }) => {
  if (!/^(initiate|download|progress|done) /.test(message)) log(logBox, message);
});

// Populate the dropdown with every catalog model that fills the `text` slot.
fillModelSelect(modelSelect, "text", "Qwen3.5-0.8B-q4f16_1-MLC");

loadBtn.addEventListener("click", async () => {
  const modelId = modelSelect.value;
  loadBtn.disabled = true;
  generateBtn.disabled = true;
  output.textContent = "";
  statsLine.textContent = "";
  try {
    log(logBox, `Loading ${modelId}…`);
    await ai.load(modelId);
    log(logBox, `Ready. Occupied slots: ${[...ai.occupiedSlots()].join(", ")}`);
    generateBtn.disabled = false;
  } catch (error) {
    if (error instanceof WebGPUUnavailableError) {
      log(logBox, "This model needs WebGPU, which this browser/device does not expose.");
    } else {
      log(logBox, `Load failed: ${(error as Error).message}`);
    }
  } finally {
    loadBtn.disabled = false;
  }
});

generateBtn.addEventListener("click", async () => {
  const messages: ChatMessage[] = [
    { role: "system", content: systemInput.value },
    { role: "user", content: promptInput.value },
  ];

  generateBtn.disabled = true;
  output.textContent = "";
  statsLine.textContent = "generating…";
  try {
    const { text, stats } = await ai.generateText(messages, {
      // onDelta receives the whole accumulated string each time — just assign it. stripThinkTags
      // hides <think>…</think> reasoning blocks (and is safe on a still-open block mid-stream).
      onDelta: (full) => {
        output.textContent = stripThinkTags(full);
      },
    });
    output.textContent = stripThinkTags(text);
    statsLine.textContent = formatStats(stats);
  } catch (error) {
    statsLine.textContent = "";
    output.textContent = `Error: ${(error as Error).message}`;
  } finally {
    generateBtn.disabled = false;
  }
});
