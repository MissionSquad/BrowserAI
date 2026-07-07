import { KOKORO_MODEL_ID, type ModelTask } from "../models.js";
import { EMPTY_RESPONSE } from "../inference.js";

/**
 * Pure helpers for the voice pipeline (mic → STT → text LLM → TTS voice assistant). Kept free of
 * engine and DOM state so the chain-readiness logic, recommended-model map, and history cap are
 * unit-testable.
 */

/** The capability slots the voice-assistant pipeline chains, in signal order (mic → … → speakers). */
export const PIPELINE_TASKS = ["stt", "text", "tts"] as const;
export type PipelineTask = (typeof PIPELINE_TASKS)[number];

/**
 * The recommended model per pipeline slot — low-resource defaults (Whisper Base / Qwen3.5 0.8B /
 * Kokoro 82M). They occupy disjoint capability slots, so loading all three never evicts another;
 * total ≈2.5 GB VRAM.
 */
export const PIPELINE_RECOMMENDED: Record<PipelineTask, string> = {
  stt: "onnx-community/whisper-base",
  text: "Qwen3.5-0.8B-q4f16_1-MLC",
  tts: KOKORO_MODEL_ID,
};

/** The pipeline slots not yet filled by a loaded model, in {@link PIPELINE_TASKS} order. */
export function missingPipelineModels(filledSlots: ReadonlySet<ModelTask>): PipelineTask[] {
  return PIPELINE_TASKS.filter((task) => !filledSlots.has(task));
}

/** Whether a generated reply is worth speaking — skips blank output and the empty-response sentinel. */
export function shouldSpeakReply(reply: string): boolean {
  const trimmed = reply.trim();
  return trimmed.length > 0 && trimmed !== EMPTY_RESPONSE;
}

/** How many user↔assistant exchanges the pipeline keeps as conversation context. */
export const PIPELINE_HISTORY_MAX_TURNS = 6;

/**
 * Cap a pipeline conversation history (alternating user/assistant entries, system turn excluded) to
 * the most recent `maxTurns` exchanges. Returns the input array unchanged when already within bounds.
 */
export function capPipelineHistory<T>(messages: T[], maxTurns: number = PIPELINE_HISTORY_MAX_TURNS): T[] {
  const maxEntries = maxTurns * 2;
  return messages.length > maxEntries ? messages.slice(-maxEntries) : messages;
}

/**
 * The newly-appended suffix when `next` extends `prev` (a monotonically-growing stream); "" when `next`
 * is not an extension of `prev` (e.g. a think-tag block was stripped, shrinking the text). Used to push
 * only new text into a streaming TTS splitter without repeating already-spoken text.
 */
export function streamedDelta(prev: string, next: string): string {
  return next.startsWith(prev) ? next.slice(prev.length) : "";
}

/** System turn for the voice assistant — replies must be short and speakable (fed to TTS verbatim). */
export const PIPELINE_SYSTEM_PROMPT =
  "You are a helpful voice assistant. Reply in 1-3 short, natural sentences suitable for being spoken aloud. Do not use markdown, lists, or code blocks.";
