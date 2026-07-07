import type { BrowserAI } from "../client.js";
import type { ChatMessage } from "../inference.js";
import type { RuntimeParameters } from "../generation.js";
import { runtimeParamsFromPreset } from "../generation.js";
import { getModelPreset } from "../models.js";
import type { TtsDefaults } from "../config.js";
import { TypedEmitter } from "../events.js";
import { formatError, MicrophoneError, VoiceTurnError } from "../errors.js";
import { AUDIO_SR, pitchShiftBuffer } from "../audio/pcm.js";
import { MicRecorder } from "../audio/recorder.js";
import type { StreamingAudioPlayer, StreamingAudioPlayerOptions } from "../audio/streaming-player.js";
import { createStreamingAudioPlayer } from "../audio/streaming-player.js";
import type { SynthClip } from "../audio/clip.js";
import { clipPlaybackRate, playClipThroughElement, releaseClip } from "../audio/clip.js";
import type { LoadProgress } from "../client.js";
import type { PipelineTask } from "./helpers.js";
import {
  capPipelineHistory,
  missingPipelineModels,
  PIPELINE_HISTORY_MAX_TURNS,
  PIPELINE_RECOMMENDED,
  PIPELINE_SYSTEM_PROMPT,
  PIPELINE_TASKS,
  shouldSpeakReply,
  streamedDelta,
} from "./helpers.js";
import { stripThinkTags } from "../text-utils.js";

export type VoicePipelineStage = "idle" | "listening" | "transcribing" | "thinking" | "speaking" | "error";

export type TurnDiscardReason = "contention" | "too-short" | "no-speech" | "empty-reply";

export type VoicePipelineEvents = {
  /** Every stage transition. Fired mid-stream too (thinking → speaking on the first audio chunk). */
  stagechange: { stage: VoicePipelineStage; previousStage: VoicePipelineStage };
  /** The mic turned on/off. `recording` can outlive "listening" briefly while the take decodes. */
  recordingchange: { recording: boolean };
  /** The mic is live — attach an AnalyserNode here for input visualization. */
  micstream: { context: AudioContext; source: MediaStreamAudioSourceNode; stream: MediaStream };
  /** The mic graph is being torn down — dispose whatever `micstream` created. */
  micstreamended: Record<string, never>;
  /**
   * Partial transcript while transcribing. Only generative STT models (Gemma, Granite Speech,
   * Voxtral) stream partials — plain Whisper/Moonshine pipelines emit nothing until the final text,
   * so keep a placeholder until the `transcript` event.
   */
  partialtranscript: { text: string };
  /** The final accepted transcript for the turn. */
  transcript: { text: string };
  /** Streaming reply text — the FULL accumulated, think-stripped text so far. */
  replydelta: { text: string };
  /** Audio is about to be heard: streamed sentences (Kokoro) or one whole clip (other TTS). */
  speechstart: { mode: "streaming" | "clip" };
  /** Non-streaming path only: the synthesized clip, before playback starts. */
  replyclip: { clip: SynthClip };
  /** The turn committed: history now includes this user/assistant exchange. */
  turncommitted: { transcript: string; reply: string; messages: readonly ChatMessage[] };
  /** The take/turn was dropped without committing (see reason). Live UI should discard partials. */
  turndiscarded: { reason: TurnDiscardReason };
  /** A component failed; the pipeline is in the "error" stage until the next take or reset. */
  error: { label: string; message: string; permissionDenied: boolean };
  /**
   * Playback fully drained. NOTE: the pipeline goes idle at synthesis-drain — the tail of playback
   * continues during idle (this keeps talk re-enabled and barge-in possible), so this event fires
   * after the turn already committed.
   */
  playbackfinished: Record<string, never>;
  /** reset() cleared the conversation. */
  reset: Record<string, never>;
};

export type VoicePipelineOptions = {
  /** System turn for the assistant. Defaults to {@link PIPELINE_SYSTEM_PROMPT}. */
  systemPrompt?: string;
  /** How many user↔assistant exchanges are sent as model context (display history is unbounded). */
  historyMaxTurns?: number;
  /**
   * Runtime overrides merged over the conversational defaults (free-form decoding, temperature 0.7,
   * 256 max tokens, unpinned seed, thinking disabled where supported).
   */
  runtime?: Partial<RuntimeParameters>;
  /** Voice/speed/pitch overrides (defaults come from the client's TTS config). */
  tts?: Partial<TtsDefaults>;
  /** Minimum decoded take length in samples; shorter takes are discarded. Default 16000/4 (~¼s). */
  minTakeSamples?: number;
  /**
   * Extra busy predicate: when it returns true at turn time, the take is discarded ("contention")
   * instead of driving engines that other host work is using. The client's own busy state is always
   * checked; use this for host-side work the client cannot see.
   */
  isExternalWorkBusy?: () => boolean;
  /** Visualization taps for the streaming audio player's graph. */
  streamingAudio?: StreamingAudioPlayerOptions;
  /**
   * Element used for whole-clip playback (non-Kokoro TTS). Defaults to a detached `new Audio()`.
   * If you attach an analyzer to it, note a media element allows exactly ONE MediaElementSource for
   * its whole life — create the analyzer once and never dispose it while the element is in use.
   */
  clipAudioElement?: HTMLAudioElement;
};

/**
 * Headless press-to-talk voice assistant: mic → STT → text LLM → TTS → speakers, as a state machine
 * with typed events. One turn per press pair: `startListening()` (press) then `finishTurn()`
 * (press again) — or `toggle()` for both. With a Kokoro speech model, replies are spoken
 * sentence-by-sentence while the text model is still generating; other TTS models speak the
 * finished reply as one clip. Pressing talk while a reply plays stops it (barge-in). A turn is
 * committed to history only on success; failures leave history untouched.
 */
export class VoicePipeline extends TypedEmitter<VoicePipelineEvents> {
  readonly #client: BrowserAI;
  readonly #options: VoicePipelineOptions;
  readonly #recorder: MicRecorder;
  #stage: VoicePipelineStage = "idle";
  #recording = false;
  #error: string | null = null;
  #transcript = "";
  #reply = "";
  #messages: ChatMessage[] = [];
  #player: StreamingAudioPlayer | null = null;
  #clip: SynthClip | null = null;
  #clipAudio: HTMLAudioElement | null = null;

  constructor(client: BrowserAI, options: VoicePipelineOptions = {}) {
    super();
    this.#client = client;
    this.#options = options;
    this.#recorder = new MicRecorder({
      onStreamReady: (graph) => this.emit("micstream", graph),
      onStreamEnded: () => this.emit("micstreamended", {}),
    });
  }

  get stage(): VoicePipelineStage {
    return this.#stage;
  }
  get recording(): boolean {
    return this.#recording;
  }
  /** Full session history (committed turns only; unbounded — the model sees a capped slice). */
  get messages(): readonly ChatMessage[] {
    return this.#messages;
  }
  get transcript(): string {
    return this.#transcript;
  }
  get reply(): string {
    return this.#reply;
  }
  get error(): string | null {
    return this.#error;
  }

  /** Mid-chain (mic stopped, models running). Listening is NOT busy. */
  busy(): boolean {
    return this.#stage === "transcribing" || this.#stage === "thinking" || this.#stage === "speaking";
  }

  /** Busy OR listening — the pipeline owns (or is about to own) the STT/text/TTS models. */
  active(): boolean {
    return this.busy() || this.#recording;
  }

  /** The chain slots (stt/text/tts) not yet filled by a loaded model. */
  missingModels(): PipelineTask[] {
    return missingPipelineModels(this.#client.occupiedSlots());
  }

  /** Load the recommended model for each unfilled chain slot, sequentially. */
  async loadRecommendedModels(onProgress?: (info: LoadProgress) => void): Promise<void> {
    for (const task of PIPELINE_TASKS) {
      if (!this.#client.slotOwner(task)) {
        await this.#client.load(PIPELINE_RECOMMENDED[task], { onProgress });
      }
    }
  }

  /**
   * Start capturing a take (press-to-talk press). Stops any still-playing reply first (barge-in).
   * Returns false without side effects when a take is already live, a turn is running, the chain
   * models are missing, or other work is busy. Mic failures land in the "error" stage.
   */
  async startListening(): Promise<boolean> {
    if (this.#recording || this.busy() || this.#otherWorkBusy()) return false;
    if (this.missingModels().length > 0) return false;
    // Barge-in: stop a still-playing reply (whole-clip element or the streaming player).
    this.#stopPlayback();
    try {
      const started = await this.#recorder.start();
      if (!started) return false;
      this.#recording = true;
      this.emit("recordingchange", { recording: true });
      this.#setStage("listening");
      return true;
    } catch (error) {
      const denied = error instanceof MicrophoneError && error.permissionDenied;
      this.#setError(
        "Microphone",
        denied ? "Microphone access was denied. Allow microphone use for this site and try again." : formatError(error),
        denied,
      );
      return false;
    }
  }

  /** Press-to-talk convenience: start a take, or finish the live one. */
  async toggle(): Promise<void> {
    if (this.#recording) await this.finishTurn();
    else await this.startListening();
  }

  /**
   * Finish the live take and run the chain: decode → transcribe → generate → speak → commit.
   * Stage/`turndiscarded`/`turncommitted`/`error` events narrate the run.
   */
  async finishTurn(): Promise<void> {
    if (this.busy()) return;
    const stt = this.#client.sttModel();
    const llm = this.#client.textModel();
    const tts = this.#client.ttsModel();
    if (!stt?.multimodal || !llm || !tts?.multimodal) {
      await this.#stopRecorder().catch(() => undefined); // still release the mic
      this.#setError("Voice pipeline", "Load the transcription, text and speech models before talking.", false);
      return;
    }

    try {
      this.#setStage("transcribing");
      const pcm = await this.#stopRecorder();
      if (this.#otherWorkBusy()) {
        // Other work started while we were listening — running the chain now would drive the same
        // engines concurrently. Discard the take.
        this.#discard("contention");
        return;
      }
      if (!pcm || pcm.length < (this.#options.minTakeSamples ?? AUDIO_SR / 4)) {
        this.#discard("too-short");
        return;
      }

      const sttResult = await this.#client.transcribe(pcm, (partial) => this.emit("partialtranscript", { text: partial }));
      const transcript = sttResult.text.trim();
      this.#transcript = transcript;
      if (!transcript) {
        this.#discard("no-speech");
        return;
      }
      this.emit("transcript", { text: transcript });

      this.#setStage("thinking");
      const preset = getModelPreset(llm.modelId);
      // Conversational overrides on the model's defaults: free-form (no JSON constraint), a bit of
      // sampling warmth, short spoken-style replies, and no thinking tokens where supported.
      const runtime: RuntimeParameters = {
        ...runtimeParamsFromPreset(preset.defaultRuntime),
        jsonMode: "none",
        temperature: 0.7,
        maxTokens: 256,
        seed: null, // extraction presets pin a seed for reproducibility; conversation should vary
        disableThinking: preset.disableThinkingSupported,
        ...this.#options.runtime,
      };
      // Hosts may display the whole session; only the most recent turns go to the model.
      const messages: ChatMessage[] = [
        { role: "system", content: this.#options.systemPrompt ?? PIPELINE_SYSTEM_PROMPT },
        ...capPipelineHistory(this.#messages, this.#options.historyMaxTurns ?? PIPELINE_HISTORY_MAX_TURNS),
        { role: "user", content: transcript },
      ];

      let reply: string;
      if (this.#client.supportsStreamingTts()) {
        // Kokoro: speak sentences as the LLM produces them (audio starts after the first sentence).
        reply = await this.#runStreamingReply(messages, runtime);
      } else {
        // Non-streaming TTS (e.g. Supertonic): whole reply, then one clip.
        const { text } = await this.#client.generateText(messages, {
          runtime,
          onDelta: (full) => this.emit("replydelta", { text: stripThinkTags(full) }),
        });
        reply = stripThinkTags(text).trim();
        if (shouldSpeakReply(reply)) {
          this.#setStage("speaking");
          this.emit("speechstart", { mode: "clip" });
          const clip = await this.#client.synthesizeToClip(reply, this.#tts());
          releaseClip(this.#clip);
          this.#clip = clip;
          this.emit("replyclip", { clip });
          await this.#playClip(clip);
        }
      }

      this.#reply = reply;
      if (!shouldSpeakReply(reply)) {
        this.#discard("empty-reply");
        return;
      }
      this.#messages = [...this.#messages, { role: "user", content: transcript }, { role: "assistant", content: reply }];
      this.emit("turncommitted", { transcript, reply, messages: this.#messages });
      this.#setStage("idle");
    } catch (error) {
      // A mid-stream failure must release the streaming player before surfacing.
      this.#player?.stop();
      this.#player = null;
      const label = error instanceof VoiceTurnError ? error.label : stageErrorLabel(this.#stage);
      this.#setError(label, formatError(error), false);
    }
  }

  /**
   * Clear the conversation (stops any playing reply). Returns false while a take/turn is active —
   * resetting mid-turn would pull history out from under the running chain.
   */
  reset(): boolean {
    if (this.active()) return false;
    this.#stopPlayback();
    releaseClip(this.#clip);
    this.#clip = null;
    this.#messages = [];
    this.#transcript = "";
    this.#reply = "";
    this.#setStage("idle");
    this.emit("reset", {});
    return true;
  }

  /** Tear the pipeline down: stop playback, abandon any live take, drop all listeners. */
  async dispose(): Promise<void> {
    this.#stopPlayback();
    releaseClip(this.#clip);
    this.#clip = null;
    // Detach from a host-supplied clip element so serial pipeline instances don't stack listeners.
    this.#clipAudio?.removeEventListener("ended", this.#clipEnded);
    this.#clipAudio = null;
    await this.#recorder.cancel().catch(() => undefined);
    if (this.#recording) {
      this.#recording = false;
      this.emit("recordingchange", { recording: false });
    }
    this.removeAllListeners();
  }

  /* ------------------------------------------------------------------ */
  /* Streaming reply (Kokoro)                                            */
  /* ------------------------------------------------------------------ */

  /**
   * Generate and speak simultaneously: LLM deltas are pushed into a Kokoro TextSplitterStream,
   * which emits complete sentences; each sentence is synthesized and scheduled gaplessly, so audio
   * starts after the FIRST sentence instead of after the whole reply. Returns the final
   * think-stripped reply. The turn goes idle at synthesis-drain — the tail of playback continues in
   * idle (like the whole-clip path, whose playback resolves at start), so talk re-enables and
   * barge-in stays possible ({@link startListening} stops the player).
   */
  async #runStreamingReply(messages: ChatMessage[], runtime: RuntimeParameters): Promise<string> {
    const tts = this.#tts();
    const session = await this.#client.createStreamingTts({ voice: tts.voice, speed: tts.speed });
    let player: StreamingAudioPlayer;
    try {
      player = createStreamingAudioPlayer(this.#options.streamingAudio);
    } catch (error) {
      // The session already holds the TTS serialization lock — release it or the client deadlocks.
      session.splitter.close();
      session.release();
      throw new VoiceTurnError("Speech synthesis failed", error);
    }
    this.#player = player;

    // Consumer: synthesize + schedule each completed sentence; flip to "speaking" on first audio.
    let heardAudio = false;
    const consumer = (async () => {
      for await (const audio of session.chunks) {
        if (this.#player !== player) break; // barge-in/error tore this player down — stop synthesizing
        if (!heardAudio) {
          heardAudio = true;
          this.#setStage("speaking");
          this.emit("speechstart", { mode: "streaming" });
        }
        // Bake the pitch per chunk (duration-preserving), matching the whole-clip path. Speed is
        // already baked into synthesis by Kokoro, so nothing plays with a playbackRate here.
        const pcm = tts.pitch === 1 ? audio.audio : pitchShiftBuffer(audio.audio, tts.pitch);
        player.push(pcm, audio.sampling_rate);
      }
    })();
    consumer.catch(() => undefined); // silences the transient gap; real handling is at the awaits below
    // The consumer always terminates (the splitter is closed on every path below, and barge-in
    // breaks the loop), and its settlement is exactly synthesis-drain — release the TTS lock there.
    void consumer.finally(() => session.release()).catch(() => undefined);

    let text: string;
    let pushed = "";
    try {
      // onDelta delivers the FULL accumulated text each call — push only the monotonic new suffix of
      // the think-stripped text. A think-tag strip can shrink the text (delta === ""); the tail push
      // below still speaks whatever the incremental pushes missed.
      ({ text } = await this.#client.generateText(messages, {
        runtime,
        onDelta: (full) => {
          const clean = stripThinkTags(full);
          this.emit("replydelta", { text: clean });
          const delta = streamedDelta(pushed, clean);
          if (delta) {
            session.splitter.push(delta);
            pushed = clean;
          }
        },
      }));
    } catch (error) {
      session.splitter.close(); // unblock the consumer's iterator so it can exit
      player.stop();
      // Tag explicitly: the consumer may already have flipped the stage to "speaking".
      throw new VoiceTurnError("Text generation failed", error);
    }

    const reply = stripThinkTags(text).trim();
    if (!shouldSpeakReply(reply)) {
      session.splitter.close();
      player.stop(); // nothing worth speaking — drop whatever was queued
      this.#player = null; // cleared before the await so the consumer's break-check stops synthesis
      await consumer.catch(() => undefined);
      return reply;
    }

    // Push whatever the incremental deltas didn't cover (e.g. after a non-monotonic think-tag strip).
    const tail = streamedDelta(pushed, stripThinkTags(text));
    if (tail) session.splitter.push(tail);
    session.splitter.close();
    try {
      await consumer; // every sentence synthesized + scheduled
    } catch (error) {
      // Tag explicitly: a synthesis failure can surface here while the stage still says "thinking".
      throw new VoiceTurnError("Speech synthesis failed", error);
    }
    // Don't await playback: the player tears itself down when the last chunk finishes, clearing
    // #player unless a newer player has already replaced it. `playbackfinished` fires only on a
    // true drain — a barge-in/reset/dispose stop() clears #player first, so the identity check
    // filters the interrupted case (stop() also resolves the parked end() promise).
    void player.end().then(() => {
      if (this.#player === player) {
        this.#player = null;
        this.emit("playbackfinished", {});
      }
    });
    return reply;
  }

  /* ------------------------------------------------------------------ */
  /* Internals                                                           */
  /* ------------------------------------------------------------------ */

  #tts(): TtsDefaults {
    return { ...this.#client.config.tts, ...this.#options.tts };
  }

  #otherWorkBusy(): boolean {
    return this.#client.isBusy() || this.#options.isExternalWorkBusy?.() === true;
  }

  async #stopRecorder(): Promise<Float32Array | null> {
    try {
      return await this.#recorder.stop();
    } finally {
      if (this.#recording) {
        this.#recording = false;
        this.emit("recordingchange", { recording: false });
      }
    }
  }

  readonly #clipEnded = (): void => this.emit("playbackfinished", {});

  async #playClip(clip: SynthClip): Promise<void> {
    if (!this.#clipAudio) {
      this.#clipAudio = this.#options.clipAudioElement ?? new Audio();
      this.#clipAudio.addEventListener("ended", this.#clipEnded);
    }
    const rate = clipPlaybackRate(this.#client.ttsModel()?.multimodal, this.#tts().speed);
    // Resolves at playback START (matching the streaming path's idle-at-synthesis-drain semantics).
    await playClipThroughElement(this.#clipAudio, clip, rate);
  }

  #stopPlayback(): void {
    this.#clipAudio?.pause();
    this.#player?.stop();
    this.#player = null;
  }

  #setStage(stage: VoicePipelineStage): void {
    const previousStage = this.#stage;
    this.#stage = stage;
    this.#error = null;
    if (stage !== previousStage) this.emit("stagechange", { stage, previousStage });
  }

  #discard(reason: TurnDiscardReason): void {
    this.emit("turndiscarded", { reason });
    this.#setStage("idle");
  }

  #setError(label: string, message: string, permissionDenied: boolean): void {
    const previousStage = this.#stage;
    this.#stage = "error";
    this.#error = `${label}: ${message}`;
    if (previousStage !== "error") this.emit("stagechange", { stage: "error", previousStage });
    this.emit("error", { label, message, permissionDenied });
  }
}

function stageErrorLabel(stage: VoicePipelineStage): string {
  if (stage === "transcribing") return "Transcription failed";
  if (stage === "thinking") return "Text generation failed";
  if (stage === "speaking") return "Speech synthesis failed";
  return "Voice pipeline failed";
}
