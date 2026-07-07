import { AUDIO_SR } from "./pcm.js";

/**
 * Decode an audio file (ArrayBuffer) to a 16 kHz mono Float32Array, matching Transformers.js
 * `read_audio` output. Browser main-thread only: uses `AudioContext`/`OfflineAudioContext`, which do
 * not exist in workers or Node — decode on the page and pass the PCM into worker-side consumers.
 */
export async function decodeAudioTo16kMono(buffer: ArrayBuffer): Promise<Float32Array> {
  const ctx = new AudioContext();
  // Close the context even when decode rejects (e.g. a header-only blob from an instant start/stop).
  const decoded = await ctx.decodeAudioData(buffer).finally(() => ctx.close().catch(() => undefined));
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * AUDIO_SR), AUDIO_SR);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}
