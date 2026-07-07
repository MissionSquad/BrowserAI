/**
 * Example 4 — Full voice assistant with VoicePipeline.
 *
 * The pipeline wraps a BrowserAI client and drives the whole mic → STT → LLM → TTS chain. You never
 * call the models directly; you press to talk and subscribe to events:
 *   stagechange      — idle | listening | transcribing | thinking | speaking | error
 *   transcript       — the accepted user utterance
 *   replydelta       — the assistant reply so far (full accumulated text)
 *   turncommitted    — the exchange was appended to history
 *   turndiscarded    — the take was dropped (too-short / no-speech / empty-reply / contention)
 *   error            — a component failed (includes permissionDenied for mic denials)
 */
import { BrowserAI, VoicePipeline } from "@missionsquad/browserai";
import { byId, setProgress } from "./shared.js";

const loadBtn = byId<HTMLButtonElement>("loadModels");
const loadStatus = byId("loadStatus");
const progress = byId<HTMLProgressElement>("progress");
const stageBadge = byId("stage");
const talkBtn = byId<HTMLButtonElement>("talk");
const resetBtn = byId<HTMLButtonElement>("reset");
const errorBox = byId("error");
const bubbles = byId("bubbles");
const live = byId("live");

const ai = new BrowserAI();
const voice = new VoicePipeline(ai);

// A placeholder bubble that shows the streaming reply until the turn commits.
let pendingReply: HTMLElement | null = null;

function addBubble(role: "user" | "assistant", text: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  el.textContent = text;
  bubbles.appendChild(el);
  bubbles.scrollTop = bubbles.scrollHeight;
  return el;
}

voice.on("stagechange", ({ stage }) => {
  stageBadge.textContent = stage;
  // "listening" starts on a press; talk is re-enabled between turns.
  talkBtn.textContent = stage === "listening" ? "■ Sending — press to stop" : "🎙 Press to talk";
});

voice.on("error", ({ label, message }) => {
  errorBox.textContent = `${label}: ${message}`;
});

voice.on("transcript", ({ text }) => {
  errorBox.textContent = "";
  addBubble("user", text);
  live.textContent = "";
  pendingReply = addBubble("assistant", "…");
});

// replydelta delivers the FULL accumulated reply each time — assign it to the live bubble.
voice.on("replydelta", ({ text }) => {
  if (pendingReply) pendingReply.textContent = text;
});

voice.on("turncommitted", () => {
  pendingReply = null; // the bubble now holds the final reply
});

voice.on("turndiscarded", ({ reason }) => {
  if (pendingReply) {
    pendingReply.remove();
    pendingReply = null;
  }
  live.textContent = `(take discarded: ${reason})`;
});

voice.on("reset", () => {
  bubbles.replaceChildren();
  live.textContent = "";
  errorBox.textContent = "";
});

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  loadStatus.textContent = "loading…";
  try {
    await voice.loadRecommendedModels((info) => setProgress(progress, loadStatus, info.progress, info.status));
    loadStatus.textContent = "ready";
    talkBtn.disabled = false;
    resetBtn.disabled = false;
  } catch (error) {
    loadStatus.textContent = `load failed: ${(error as Error).message}`;
    loadBtn.disabled = false;
  }
});

// toggle() = start listening on the first press, finish and run the turn on the second.
talkBtn.addEventListener("click", () => {
  void voice.toggle();
});

resetBtn.addEventListener("click", () => {
  voice.reset();
});
