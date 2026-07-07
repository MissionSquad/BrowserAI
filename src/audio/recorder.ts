import { MicrophoneError } from "../errors.js";
import { decodeAudioTo16kMono } from "./decode.js";

/**
 * Press-to-talk microphone recorder: getUserMedia → MediaRecorder with no timeslice (one blob at
 * stop) → 16 kHz mono PCM. Hardened lifecycle ported from small-ai's pipeline recorder:
 * - a synchronous pending claim so a double-press cannot start two recorders;
 * - an inactive-recorder guard (the recorder auto-stops when its track ends — device unplugged,
 *   permission revoked — and `stop()` would then be a silent no-op whose `onstop` never fires);
 * - try/finally teardown that always releases the mic tracks and closes the AudioContext.
 *
 * The recorder creates an AudioContext + MediaStreamSource so hosts can visualize input levels:
 * `onStreamReady` receives them (attach an AnalyserNode there). The SDK never renders anything.
 */
export type MicRecorderOptions = {
  /** Called once per take after the mic is live — attach analyzers/visualizers here. */
  onStreamReady?: (graph: { context: AudioContext; source: MediaStreamAudioSourceNode; stream: MediaStream }) => void;
  /** Called during teardown, before the context closes — dispose whatever onStreamReady created. */
  onStreamEnded?: () => void;
};

type ActiveTake = {
  ctx: AudioContext;
  stream: MediaStream;
  recorder: MediaRecorder;
  chunks: Blob[];
};

export class MicRecorder {
  readonly #options: MicRecorderOptions;
  #pending = false;
  #abortPending = false;
  #take: ActiveTake | null = null;

  constructor(options: MicRecorderOptions = {}) {
    this.#options = options;
  }

  /** Whether a take is currently being captured (true between a successful start() and stop()). */
  get recording(): boolean {
    return this.#take !== null;
  }

  #streamEnded(): void {
    try {
      this.#options.onStreamEnded?.();
    } catch {
      // A throwing host disposal hook must not interrupt mic/context teardown.
    }
  }

  /**
   * Start capturing. No-op (returns false) when already recording or a start is mid-flight.
   * @throws {MicrophoneError} when the mic cannot be acquired (`permissionDenied` set for denials).
   */
  async start(): Promise<boolean> {
    if (this.#pending || this.#take) return false;
    this.#pending = true; // claimed synchronously before the getUserMedia await (double-press guard)
    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (this.#abortPending) {
        // cancel() ran while the permission prompt / acquisition was pending — release and bail.
        this.#abortPending = false;
        stream.getTracks().forEach((track) => track.stop());
        return false;
      }
      ctx = new AudioContext();
      const source = ctx.createMediaStreamSource(stream);
      try {
        this.#options.onStreamReady?.({ context: ctx, source, stream });
      } catch {
        // A throwing host visualization hook must not abort the take.
      }
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (event) => chunks.push(event.data);
      recorder.start();
      this.#take = { ctx, stream, recorder, chunks };
      return true;
    } catch (error) {
      // Release anything acquired before the failure — otherwise the mic indicator stays lit.
      stream?.getTracks().forEach((track) => track.stop());
      this.#streamEnded();
      if (ctx) void ctx.close().catch(() => undefined);
      throw new MicrophoneError(error);
    } finally {
      this.#pending = false;
    }
  }

  /**
   * Stop the take and decode it to 16 kHz mono PCM. Returns null when nothing was recorded (no
   * take in flight, or an empty capture). Always releases the mic, even if stopping throws.
   */
  async stop(): Promise<Float32Array | null> {
    const take = this.#take;
    this.#take = null; // claim synchronously so a re-entrant stop can't double-clean
    if (!take) return null;
    const { recorder, stream, ctx, chunks } = take;
    try {
      await new Promise<void>((resolve) => {
        // The recorder auto-stops when its track ends (device unplugged, permission revoked). stop()
        // would then be a silent no-op and onstop would never fire — resolve immediately instead.
        if (recorder.state === "inactive") {
          resolve();
          return;
        }
        recorder.onstop = () => resolve();
        recorder.stop();
      });
    } finally {
      // Always release the mic and context — even if stop() threw (legacy InvalidStateError).
      stream.getTracks().forEach((track) => track.stop());
      this.#streamEnded();
      await ctx.close().catch(() => undefined);
    }
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    if (blob.size === 0) return null;
    return decodeAudioTo16kMono(await blob.arrayBuffer());
  }

  /** Abandon a take without decoding it (releases the mic; the audio is discarded). Also aborts a start() still awaiting the permission prompt. */
  async cancel(): Promise<void> {
    if (this.#pending && !this.#take) this.#abortPending = true; // start() releases the mic when getUserMedia resolves
    const take = this.#take;
    this.#take = null;
    if (!take) return;
    const { recorder, stream, ctx } = take;
    try {
      await new Promise<void>((resolve) => {
        if (recorder.state === "inactive") {
          resolve();
          return;
        }
        recorder.onstop = () => resolve();
        recorder.stop();
      });
    } finally {
      stream.getTracks().forEach((track) => track.stop());
      this.#streamEnded();
      await ctx.close().catch(() => undefined);
    }
  }
}
