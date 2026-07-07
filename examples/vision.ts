/**
 * Example 5 — Vision question answering.
 *
 * Load a VLM into the vision slot, then call `describeImage(imageSrc, prompt, onDelta)`. The image
 * source can be a data URL (as here, read from a file) or an https URL the browser can fetch. The
 * result is a VlmResult: the answer text plus token/timing fields.
 */
import { BrowserAI, WebGPUUnavailableError } from "@missionsquad/browserai";
import { byId, fillModelSelect, log, setProgress } from "./shared.js";

const modelSelect = byId<HTMLSelectElement>("model");
const loadBtn = byId<HTMLButtonElement>("load");
const askBtn = byId<HTMLButtonElement>("ask");
const progress = byId<HTMLProgressElement>("progress");
const progressLabel = byId("progressLabel");
const logBox = byId("log");
const fileInput = byId<HTMLInputElement>("file");
const preview = byId<HTMLImageElement>("preview");
const questionInput = byId<HTMLInputElement>("question");
const answerOut = byId("answer");
const statsLine = byId("stats");

const ai = new BrowserAI();
ai.on("loadprogress", ({ progress: p, status }) => setProgress(progress, progressLabel, p, status));
// Skip the per-chunk "initiate/download/progress/done <file>" statuses Transformers.js models emit.
ai.on("status", ({ message }) => {
  if (!/^(initiate|download|progress|done) /.test(message)) log(logBox, message);
});

fillModelSelect(modelSelect, "vision", "HuggingFaceTB/SmolVLM-256M-Instruct");

let modelReady = false;
let imageDataUrl: string | null = null;

function refreshAskButton(): void {
  askBtn.disabled = !(modelReady && imageDataUrl);
}

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  modelReady = false;
  refreshAskButton();
  try {
    log(logBox, `Loading ${modelSelect.value}…`);
    await ai.load(modelSelect.value);
    log(logBox, "Ready.");
    modelReady = true;
  } catch (error) {
    log(logBox, error instanceof WebGPUUnavailableError ? "Vision models require WebGPU, which is unavailable here." : `Load failed: ${(error as Error).message}`);
  } finally {
    loadBtn.disabled = false;
    refreshAskButton();
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    imageDataUrl = typeof reader.result === "string" ? reader.result : null;
    if (imageDataUrl) {
      preview.src = imageDataUrl;
      preview.hidden = false;
    }
    refreshAskButton();
  };
  reader.readAsDataURL(file);
});

askBtn.addEventListener("click", async () => {
  if (!imageDataUrl) return;
  askBtn.disabled = true;
  answerOut.textContent = "";
  statsLine.textContent = "thinking…";
  try {
    const result = await ai.describeImage(imageDataUrl, questionInput.value, (partial) => {
      answerOut.textContent = partial;
    });
    answerOut.textContent = result.text;
    const elapsedMs = result.endedMs - result.startedMs;
    statsLine.textContent = `${result.completionTokens} tokens · ${(elapsedMs / 1000).toFixed(2)} s`;
  } catch (error) {
    statsLine.textContent = "";
    answerOut.textContent = `Error: ${(error as Error).message}`;
  } finally {
    askBtn.disabled = false;
  }
});
