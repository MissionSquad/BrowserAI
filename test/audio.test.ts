import { describe, expect, it } from "vitest";
import { AUDIO_SR, encodeWavBlob, pitchShiftBuffer } from "../src/audio/pcm.js";
import { planGemmaAudioWindows, supertonicVoiceUrl } from "../src/multimodal.js";
import { KOKORO_MODEL_ID, SUPERTONIC_MODEL_ID, ttsVoicesForModel } from "../src/models.js";

/** Estimate the fundamental frequency of a (near-)sine buffer via autocorrelation. */
function detectFrequency(buffer: Float32Array, sampleRate: number): number {
  const minLag = Math.floor(sampleRate / 2000); // up to 2 kHz
  const maxLag = Math.floor(sampleRate / 80); // down to 80 Hz
  let bestLag = minLag;
  let best = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let sum = 0;
    for (let i = 0; i + lag < buffer.length; i += 1) sum += buffer[i] * buffer[i + lag];
    if (sum > best) {
      best = sum;
      bestLag = lag;
    }
  }
  return sampleRate / bestLag;
}

function sine(freq: number, sampleRate: number, seconds: number): Float32Array {
  const out = new Float32Array(Math.floor(sampleRate * seconds));
  for (let i = 0; i < out.length; i += 1) out[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate);
  return out;
}

const CHUNK = 30 * AUDIO_SR; // 30s window
const MAX = 12 * 60 * AUDIO_SR; // 12min ceiling

describe("planGemmaAudioWindows", () => {
  it("returns no windows for empty audio", () => {
    expect(planGemmaAudioWindows(0)).toEqual([]);
  });

  it("keeps a sub-30s clip as one window covering the whole clip", () => {
    expect(planGemmaAudioWindows(10 * AUDIO_SR)).toEqual([{ start: 0, end: 10 * AUDIO_SR }]);
  });

  it("splits a 90s clip into three contiguous 30s windows", () => {
    expect(planGemmaAudioWindows(90 * AUDIO_SR)).toEqual([
      { start: 0, end: CHUNK },
      { start: CHUNK, end: 2 * CHUNK },
      { start: 2 * CHUNK, end: 3 * CHUNK },
    ]);
  });

  it("truncates clips longer than the 12-minute ceiling", () => {
    const overLong = planGemmaAudioWindows(13 * 60 * AUDIO_SR);
    expect(overLong).toEqual(planGemmaAudioWindows(MAX));
    expect(overLong[overLong.length - 1].end).toBe(MAX);
  });

  it("produces gap-free, non-overlapping windows that tile the usable range", () => {
    const windows = planGemmaAudioWindows(95 * AUDIO_SR);
    expect(windows[0].start).toBe(0);
    for (let i = 1; i < windows.length; i += 1) {
      expect(windows[i].start).toBe(windows[i - 1].end);
      expect(windows[i].end).toBeGreaterThan(windows[i].start);
    }
    expect(windows[windows.length - 1].end).toBe(95 * AUDIO_SR);
  });
});

describe("ttsVoicesForModel", () => {
  it("returns the 6 Kokoro voices for the Kokoro model", () => {
    expect(ttsVoicesForModel(KOKORO_MODEL_ID)).toHaveLength(6);
  });

  it("returns the 10 Supertonic voices for the Supertonic model", () => {
    const voices = ttsVoicesForModel(SUPERTONIC_MODEL_ID);
    expect(voices).toHaveLength(10);
    expect(voices.map((v) => v.id)).toEqual(["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]);
  });

  it("returns no voices for a non-TTS model", () => {
    expect(ttsVoicesForModel("Qwen3.5-0.8B-q4f16_1-MLC")).toEqual([]);
  });
});

describe("supertonicVoiceUrl", () => {
  it("builds the speaker-embedding URL against the given model's repo", () => {
    expect(supertonicVoiceUrl("onnx-community/Supertonic-TTS-ONNX", "M3")).toBe(
      "https://huggingface.co/onnx-community/Supertonic-TTS-ONNX/resolve/main/voices/M3.bin",
    );
    expect(supertonicVoiceUrl("onnx-community/Supertonic-TTS-2-ONNX", "F4")).toBe(
      "https://huggingface.co/onnx-community/Supertonic-TTS-2-ONNX/resolve/main/voices/F4.bin",
    );
  });

  it("falls back to F1 for a missing or invalid voice id", () => {
    expect(supertonicVoiceUrl("onnx-community/Supertonic-TTS-ONNX")).toMatch(/\/voices\/F1\.bin$/);
    expect(supertonicVoiceUrl("onnx-community/Supertonic-TTS-ONNX", "af_heart")).toMatch(/\/voices\/F1\.bin$/);
  });
});

describe("pitchShiftBuffer", () => {
  const SR = 24000;

  it("returns the input unchanged at ratio 1", () => {
    const input = sine(300, SR, 0.2);
    expect(pitchShiftBuffer(input, 1)).toBe(input);
  });

  it("preserves the buffer length (duration) when shifting", () => {
    const input = sine(300, SR, 0.5);
    expect(pitchShiftBuffer(input, 1.5)).toHaveLength(input.length);
    expect(pitchShiftBuffer(input, 0.5)).toHaveLength(input.length);
  });

  it("raises the fundamental frequency by the ratio", () => {
    const input = sine(300, SR, 1);
    const up = pitchShiftBuffer(input, 2);
    expect(detectFrequency(up, SR)).toBeGreaterThan(300 * 2 * 0.85);
    expect(detectFrequency(up, SR)).toBeLessThan(300 * 2 * 1.15);
  });

  it("lowers the fundamental frequency by the ratio", () => {
    const input = sine(600, SR, 1);
    const down = pitchShiftBuffer(input, 0.5);
    expect(detectFrequency(down, SR)).toBeGreaterThan(600 * 0.5 * 0.85);
    expect(detectFrequency(down, SR)).toBeLessThan(600 * 0.5 * 1.15);
  });
});

describe("encodeWavBlob", () => {
  it("produces a mono 16-bit PCM WAV blob of the expected size", async () => {
    const samples = sine(440, 24000, 0.05);
    const blob = encodeWavBlob(samples, 24000);
    expect(blob.type).toBe("audio/wav");
    expect(blob.size).toBe(44 + samples.length * 2);
    const header = new Uint8Array(await blob.arrayBuffer(), 0, 4);
    expect(String.fromCharCode(...header)).toBe("RIFF");
  });
});
