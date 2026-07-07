export type StreamingAudioPlayer = {
  /** Schedule one PCM chunk to play back-to-back after everything already scheduled. */
  push(samples: Float32Array, sampleRate: number): void;
  /** No more chunks; resolves (and tears down) once everything scheduled has finished playing. */
  end(): Promise<void>;
  /** Barge-in/error: stop playback now and tear down. Idempotent. */
  stop(): void;
};

export type StreamingAudioPlayerOptions = {
  /**
   * Called once with the player's audio graph so hosts can tap it for visualization (e.g. connect an
   * AnalyserNode to `gain`). The audible path is gain → destination only — do not connect the tap to
   * the destination or audio doubles.
   */
  onGraphReady?: (graph: { context: AudioContext; gain: GainNode }) => void;
  /** Called during teardown, before the context closes — dispose whatever onGraphReady created. */
  onTeardown?: () => void;
};

/**
 * Gapless player for streamed TTS: schedules each chunk's PCM into one AudioContext end-to-end.
 * Chunks are butt-joined at a running cursor; if synthesis falls behind playback this yields a small
 * silent gap rather than overlap. Teardown (close ctx) is idempotent, and `stop()` resolves any
 * in-flight `end()` promise so a barge-in can never hang a waiting turn.
 */
export function createStreamingAudioPlayer(options: StreamingAudioPlayerOptions = {}): StreamingAudioPlayer {
  const ctx = new AudioContext();
  const gain = ctx.createGain();
  gain.connect(ctx.destination);
  try {
    options.onGraphReady?.({ context: ctx, gain });
  } catch {
    // A throwing host visualization tap must not break (or leak) the player.
  }

  let nextTime = ctx.currentTime;
  let pending = 0;
  let ended = false;
  let torn = false;
  let onDrained: (() => void) | null = null;

  const teardown = (): void => {
    if (torn) return;
    torn = true;
    try {
      options.onTeardown?.();
    } catch {
      // Host tap disposal errors must not prevent the context from closing.
    }
    void ctx.close().catch(() => undefined);
  };
  const maybeDrain = (): void => {
    if (ended && pending === 0) {
      teardown();
      onDrained?.();
      onDrained = null;
    }
  };

  return {
    push(samples, sampleRate) {
      if (torn || samples.length === 0) return;
      // The context is often created outside a click call stack — resume defensively (autoplay policy).
      if (ctx.state === "suspended") void ctx.resume().catch(() => undefined);
      const buffer = ctx.createBuffer(1, samples.length, sampleRate);
      buffer.getChannelData(0).set(samples);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(gain);
      // If synthesis is slower than playback this yields a small silent gap rather than overlap.
      const startAt = Math.max(nextTime, ctx.currentTime);
      pending += 1;
      source.onended = () => {
        pending -= 1;
        maybeDrain();
      };
      source.start(startAt);
      nextTime = startAt + buffer.duration;
    },
    end() {
      ended = true;
      return new Promise<void>((resolve) => {
        if (pending === 0 || torn) {
          teardown();
          resolve();
          return;
        }
        onDrained = resolve;
      });
    },
    stop() {
      ended = true;
      teardown();
      onDrained?.();
      onDrained = null;
    },
  };
}
