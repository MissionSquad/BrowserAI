/**
 * Tiny DOM helpers shared by the examples. Nothing here is part of the SDK — it exists only to keep
 * each example focused on the @missionsquad/browserai API rather than on element wiring.
 */
import { MODEL_PRESETS, modelSlots } from "@missionsquad/browserai";
import type { ModelTask, RuntimeStats } from "@missionsquad/browserai";

/** Get a required element by id, narrowing to `T`. Throws if the markup is missing it. */
export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id} in the page markup`);
  return el as T;
}

/** Append a timestamped line to a log box and keep it scrolled to the bottom. */
export function log(box: HTMLElement, message: string): void {
  const time = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.textContent = `${time}  ${message}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

/** Drive a <progress> element (0–1) with an optional status label. */
export function setProgress(bar: HTMLProgressElement, label: HTMLElement, progress: number, status?: string): void {
  const value = Math.max(0, Math.min(1, Number.isFinite(progress) ? progress : 0));
  bar.value = value;
  const pct = Math.round(value * 100);
  label.textContent = status ? `${pct}% · ${status}` : `${pct}%`;
}

/** Render the fields of a RuntimeStats that most examples care about as a single line. */
export function formatStats(stats: RuntimeStats): string {
  const bits: string[] = [stats.model, stats.backend];
  if (typeof stats.completionTokens === "number") bits.push(`${stats.completionTokens} tokens`);
  // WebLLM reports precise engine telemetry in `extra`; Transformers.js fills the measured field.
  const decode = stats.extra?.decode_tokens_per_s ?? stats.measuredCompletionTokensPerSecond;
  if (typeof decode === "number") bits.push(`${decode.toFixed(1)} tok/s`);
  const ttftSec = stats.extra?.time_to_first_token_s;
  if (typeof ttftSec === "number") bits.push(`${Math.round(ttftSec * 1000)} ms TTFT`);
  bits.push(`${(stats.measuredElapsedMs / 1000).toFixed(2)} s`);
  if (stats.finishReason) bits.push(`finish: ${stats.finishReason}`);
  return bits.join("  ·  ");
}

/**
 * Populate a <select> with every catalog model that occupies `slot` (text / vision / stt / tts),
 * labelled with its short name, download size and stability rating. Selects `defaultId` if present.
 */
export function fillModelSelect(select: HTMLSelectElement, slot: ModelTask, defaultId?: string): void {
  select.replaceChildren();
  for (const preset of MODEL_PRESETS) {
    if (!modelSlots(preset).includes(slot)) continue;
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = `${preset.shortName} — ${preset.approximateDownload} — ${preset.stability}`;
    if (preset.id === defaultId) option.selected = true;
    select.appendChild(option);
  }
}
