/**
 * Custom error classes for the SDK. Every error thrown by BrowserAI code is (or wraps into) a
 * `BrowserAIError` subclass so hosts can branch with `instanceof` instead of parsing messages.
 */

/** Format an unknown thrown value as a human-readable message. */
export function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Base class for all SDK errors. */
export class BrowserAIError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** A model id that is not present in the catalog. */
export class UnknownModelError extends BrowserAIError {
  readonly modelId: string;
  constructor(modelId: string) {
    super(`Unknown model id: ${modelId}`);
    this.modelId = modelId;
  }
}

/** WebGPU is required for the requested load but is not available. */
export class WebGPUUnavailableError extends BrowserAIError {
  /** The probe's explanation (e.g. "navigator.gpu is not available."). */
  readonly reason: string;
  constructor(reason: string) {
    super(`WebGPU unavailable: ${reason}`);
    this.reason = reason;
  }
}

/** A model load failed. Wraps the underlying engine/network error as `cause`. */
export class ModelLoadError extends BrowserAIError {
  readonly modelId: string;
  constructor(modelId: string, cause: unknown) {
    super(`Model load failed for ${modelId}: ${formatError(cause)}`, { cause });
    this.modelId = modelId;
  }
}

/** The same-origin proxy did not pass its pre-load health/probe checks. */
export class ProxyVerificationError extends BrowserAIError {}

/** An operation needs a capability slot (text/vision/stt/tts) that no loaded model fills. */
export class MissingModelError extends BrowserAIError {
  readonly slot: string;
  constructor(slot: string, message?: string) {
    super(message ?? `No loaded model fills the "${slot}" capability slot.`);
    this.slot = slot;
  }
}

/** Microphone acquisition failed. `permissionDenied` distinguishes user/browser denial. */
export class MicrophoneError extends BrowserAIError {
  readonly permissionDenied: boolean;
  constructor(cause: unknown) {
    super(`Microphone error: ${formatError(cause)}`, { cause });
    this.permissionDenied =
      cause instanceof DOMException && (cause.name === "NotAllowedError" || cause.name === "SecurityError");
  }
}

/**
 * A voice-pipeline turn failure tagged with the component that failed. The streaming path runs
 * generation and synthesis CONCURRENTLY, so the pipeline stage no longer identifies the failing
 * component (the stage can be "speaking" when generation throws, or "thinking" when synthesis
 * rejects) — the throw site knows and tags the label.
 */
export class VoiceTurnError extends BrowserAIError {
  readonly label: string;
  constructor(label: string, cause: unknown) {
    super(formatError(cause), { cause });
    this.label = label;
  }
}
