/** Sample rate every STT/audio-LLM recipe in the SDK consumes (Whisper-style 16 kHz mono). */
export const AUDIO_SR = 16000;

/**
 * Pitch-shift a mono PCM buffer by `ratio` (2 = up one octave, 0.5 = down one octave) while preserving
 * duration. Time-domain granular overlap-add: Hann-windowed grains hop through the output at a fixed
 * rate (so timing is unchanged) while each grain is read from the input at `ratio×` speed via linear
 * interpolation (so pitch scales). A pure Float32Array→Float32Array function — no Web Audio — so pitch
 * can be applied offline (baked into a clip) and unit-tested independently of a browser.
 */
export function pitchShiftBuffer(input: Float32Array, ratio: number): Float32Array {
  if (ratio === 1 || input.length === 0) return input;
  const grain = 1024;
  const hop = grain / 4; // 75% overlap
  const output = new Float32Array(input.length);
  const weight = new Float32Array(input.length);
  const hann = new Float32Array(grain);
  for (let i = 0; i < grain; i += 1) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (grain - 1));

  for (let out = 0; out + grain <= output.length; out += hop) {
    for (let i = 0; i < grain; i += 1) {
      const src = out + i * ratio; // grain anchored at `out` (time preserved), read at ratio× (pitch)
      const i0 = Math.floor(src);
      if (i0 + 1 >= input.length) break;
      const frac = src - i0;
      const sample = input[i0] * (1 - frac) + input[i0 + 1] * frac;
      output[out + i] += sample * hann[i];
      weight[out + i] += hann[i];
    }
  }
  for (let i = 0; i < output.length; i += 1) if (weight[i] > 1e-6) output[i] /= weight[i]; // OLA gain normalization
  return output;
}

/** Encode a mono Float32 PCM buffer as a 16-bit PCM WAV Blob (used when pitch is baked into a clip). */
export function encodeWavBlob(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, text: string): void => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}
