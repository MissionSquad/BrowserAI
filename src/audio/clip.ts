import type { MultimodalHandle, RawAudio } from "../multimodal.js";
import { synthesize } from "../multimodal.js";
import { encodeWavBlob, pitchShiftBuffer } from "./pcm.js";

/** A synthesized, playable speech clip. `samples` is the ORIGINAL (un-pitched) PCM. */
export type SynthClip = {
  /** Object URL for the (pitch-baked) blob; revoke it when replacing the clip. Absent when `createUrl: false`. */
  url: string | null;
  blob: Blob;
  /** Original un-pitched PCM, kept so a pitch change can re-render without re-running the model. */
  samples: Float32Array;
  sampleRate: number;
  durationSec: number;
};

export type SynthesizeClipOptions = {
  voice: string;
  speed?: number;
  /** Pitch ratio baked into the blob (duration-preserving). Default 1 (blob is the model's own output). */
  pitch?: number;
  /** Create an object URL for the blob (default true). Pass false when only PCM/Blob is needed. */
  createUrl?: boolean;
};

/**
 * Synthesize `text` with a loaded TTS handle and package it as a playable clip. The pitch shift is
 * baked into the blob (duration-preserving) while `samples` keeps the original PCM so a pitch slider
 * can re-shift without re-synthesis (see {@link repitchClip}).
 */
export async function synthesizeClip(handle: MultimodalHandle, text: string, options: SynthesizeClipOptions): Promise<SynthClip> {
  const audio = await synthesize(handle, text, { voice: options.voice, speed: options.speed ?? 1 });
  return clipFromRawAudio(audio, options.pitch ?? 1, options.createUrl ?? true);
}

/** Package an already-synthesized RawAudio as a clip (same pitch-baking rules as synthesizeClip). */
export function clipFromRawAudio(audio: RawAudio, pitch: number, createUrl = true): SynthClip {
  const samples = audio.audio;
  const sampleRate = audio.sampling_rate;
  const shifted = pitchShiftBuffer(samples, pitch);
  const blob = pitch === 1 ? audio.toBlob() : encodeWavBlob(shifted, sampleRate);
  return {
    url: createUrl ? URL.createObjectURL(blob) : null,
    blob,
    samples,
    sampleRate,
    durationSec: shifted.length / sampleRate,
  };
}

/**
 * Re-render a clip from its stored original PCM at a new pitch, revoking the previous object URL.
 * Lets a pitch control take effect immediately on an existing clip without re-running the model.
 */
export function repitchClip(clip: SynthClip, pitch: number): SynthClip {
  const shifted = pitchShiftBuffer(clip.samples, pitch);
  const blob = encodeWavBlob(shifted, clip.sampleRate);
  if (clip.url) URL.revokeObjectURL(clip.url);
  return {
    ...clip,
    url: clip.url === null ? null : URL.createObjectURL(blob),
    blob,
    durationSec: shifted.length / clip.sampleRate,
  };
}

/** Revoke a clip's object URL (call when discarding or replacing a clip). Safe on URL-less clips. */
export function releaseClip(clip: SynthClip | null | undefined): void {
  if (clip?.url) URL.revokeObjectURL(clip.url);
}

/**
 * Playback rate that applies the speed setting exactly once per engine: Kokoro bakes speed into
 * synthesis (the audio is already time-scaled), so it plays at 1×; the Supertonic pipeline ignores
 * generation-time speed, so the speed setting is applied at playback instead.
 */
export function clipPlaybackRate(handle: MultimodalHandle | null | undefined, speed: number): number {
  return handle?.kind === "tts-kokoro" ? 1 : speed;
}

/**
 * Play a clip through an HTMLAudioElement. Resolves when playback STARTS (autoplay rejection is
 * swallowed), not when it ends — await the element's "ended" event for drain semantics.
 *
 * Visualization caveat for hosts: a media element can be captured by exactly ONE
 * MediaElementAudioSourceNode for its entire life, and closing the AudioContext that source routes
 * through mutes the element permanently. If you attach an analyzer to `audioEl`, create it once per
 * element and never dispose it (or its context) while the element is still in use.
 */
export async function playClipThroughElement(audioEl: HTMLAudioElement, clip: SynthClip, playbackRate = 1): Promise<void> {
  if (!clip.url) throw new Error("Clip has no object URL (synthesized with createUrl: false).");
  audioEl.src = clip.url;
  audioEl.playbackRate = playbackRate;
  await audioEl.play().catch(() => undefined);
}
