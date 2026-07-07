/**
 * Example 3 — Speech in and out.
 *
 * Transcription: `MicRecorder.start()` / `.stop()` capture the mic and decode it to 16 kHz mono PCM
 * (a Float32Array), which `BrowserAI.transcribe()` runs through the loaded stt-slot model.
 *
 * Synthesis: `BrowserAI.synthesizeToClip()` produces a `SynthClip` whose `.url` is an object URL you
 * can drop straight onto an <audio> element. Release the previous clip's URL before replacing it.
 */
import { BrowserAI, MicRecorder, MicrophoneError, releaseClip, ttsVoicesForModel, WebGPUUnavailableError } from "@missionsquad/browserai";
import type { SynthClip } from "@missionsquad/browserai";
import { byId, fillModelSelect, setProgress } from "./shared.js";

const ai = new BrowserAI();

/* ------------------------------- STT ------------------------------- */

const sttModel = byId<HTMLSelectElement>("sttModel");
const loadSttBtn = byId<HTMLButtonElement>("loadStt");
const recordBtn = byId<HTMLButtonElement>("record");
const sttProgress = byId<HTMLProgressElement>("sttProgress");
const sttStatus = byId("sttStatus");
const transcriptOut = byId("transcript");

const recorder = new MicRecorder();
let recording = false;

fillModelSelect(sttModel, "stt", "onnx-community/whisper-base");

loadSttBtn.addEventListener("click", async () => {
  loadSttBtn.disabled = true;
  sttStatus.textContent = "loading…";
  try {
    await ai.load(sttModel.value, { onProgress: (info) => setProgress(sttProgress, sttStatus, info.progress, info.status) });
    sttStatus.textContent = "ready — press Record and speak";
    recordBtn.disabled = false;
  } catch (error) {
    sttStatus.textContent = error instanceof WebGPUUnavailableError ? "WebGPU unavailable" : `load failed: ${(error as Error).message}`;
  } finally {
    loadSttBtn.disabled = false;
  }
});

recordBtn.addEventListener("click", async () => {
  if (!recording) {
    const started = await recorder.start().catch((error) => {
      const denied = error instanceof MicrophoneError && error.permissionDenied;
      sttStatus.textContent = denied ? "microphone permission denied" : `mic error: ${(error as Error).message}`;
      return false;
    });
    if (!started) return;
    recording = true;
    recordBtn.textContent = "■ Stop";
    sttStatus.textContent = "listening…";
    return;
  }

  // Stop → decode to PCM → transcribe.
  recording = false;
  recordBtn.textContent = "● Record";
  recordBtn.disabled = true;
  sttStatus.textContent = "transcribing…";
  try {
    const pcm = await recorder.stop();
    if (!pcm) {
      sttStatus.textContent = "nothing recorded";
      return;
    }
    const { text } = await ai.transcribe(pcm);
    transcriptOut.textContent = text.trim() || "(no speech detected)";
    sttStatus.textContent = "done";
  } catch (error) {
    sttStatus.textContent = `error: ${(error as Error).message}`;
  } finally {
    recordBtn.disabled = false;
  }
});

/* ------------------------------- TTS ------------------------------- */

const ttsModel = byId<HTMLSelectElement>("ttsModel");
const loadTtsBtn = byId<HTMLButtonElement>("loadTts");
const speakBtn = byId<HTMLButtonElement>("speak");
const ttsProgress = byId<HTMLProgressElement>("ttsProgress");
const ttsStatus = byId("ttsStatus");
const sayInput = byId<HTMLInputElement>("say");
const player = byId<HTMLAudioElement>("player");

let lastClip: SynthClip | null = null;
// The id of the TTS model actually loaded — the dropdown can be changed after Load without
// reloading, and Kokoro/Supertonic voice rosters are disjoint, so never derive the voice from
// the select's current value.
let loadedTtsModelId: string | null = null;

fillModelSelect(ttsModel, "tts", "onnx-community/Kokoro-82M-v1.0-ONNX");

loadTtsBtn.addEventListener("click", async () => {
  const modelId = ttsModel.value;
  loadTtsBtn.disabled = true;
  ttsStatus.textContent = "loading…";
  try {
    await ai.load(modelId, { onProgress: (info) => setProgress(ttsProgress, ttsStatus, info.progress, info.status) });
    loadedTtsModelId = modelId;
    ttsStatus.textContent = "ready";
    speakBtn.disabled = false;
  } catch (error) {
    ttsStatus.textContent = error instanceof WebGPUUnavailableError ? "WebGPU unavailable" : `load failed: ${(error as Error).message}`;
  } finally {
    loadTtsBtn.disabled = false;
  }
});

speakBtn.addEventListener("click", async () => {
  speakBtn.disabled = true;
  ttsStatus.textContent = "synthesizing…";
  try {
    // Pick a valid voice for whichever TTS model is loaded (Kokoro: "af_heart" …; Supertonic: "F1" …).
    const voice = loadedTtsModelId ? ttsVoicesForModel(loadedTtsModelId)[0]?.id : undefined;
    const clip = await ai.synthesizeToClip(sayInput.value, voice ? { voice } : {});
    releaseClip(lastClip); // revoke the previous object URL before replacing it
    lastClip = clip;
    if (clip.url) {
      player.src = clip.url;
      await player.play().catch(() => undefined);
    }
    ttsStatus.textContent = `done · ${clip.durationSec.toFixed(1)}s`;
  } catch (error) {
    ttsStatus.textContent = `error: ${(error as Error).message}`;
  } finally {
    speakBtn.disabled = false;
  }
});
