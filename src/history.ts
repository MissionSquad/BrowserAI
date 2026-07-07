import type { ModelPreset } from "./models.js";
import type { StorageLike } from "./config.js";

/**
 * Run-history persistence + aggregation. Ported from small-ai with the storage injected: pass any
 * `StorageLike` (or `null` for in-memory only); the default is `globalThis.localStorage` when present.
 */

/** Suggested task keys for logged runs (hosts may use their own strings). */
export type RunCaseKey = "json" | "chat" | "summarize" | "classify" | "qa" | "voice" | "vision";

export type ModelVendor =
  | "qwen"
  | "google"
  | "meta"
  | "openai"
  | "ibm"
  | "microsoft"
  | "nvidia"
  | "liquid"
  | "huggingface"
  | "mistral"
  | "other";

/**
 * One logged generation. This is the persisted shape, so every field is a primitive and the
 * set is intentionally minimal: exactly what comparison views need to rebuild their aggregates
 * (decode/ttft by model, run log) without storing full prompts/outputs. Storing raw text or
 * per-token arrays would blow up the storage footprint, so those are excluded.
 */
export type RunRecord = {
  /** Epoch milliseconds. */
  ts: number;
  /** Model short name, e.g. "Qwen3.5 0.8B". */
  model: string;
  /** Derived vendor bucket for filtering/labels. */
  vendor: ModelVendor;
  /** Backend used for the run. */
  backend: "webllm" | "transformers-js";
  /** Task key ({@link RunCaseKey} suggested; any non-empty string accepted). */
  case: string;
  promptTokens: number;
  completionTokens: number;
  /** Decode speed in tokens/sec. */
  decode: number;
  /** Time to first token in seconds. */
  ttft: number;
  /** Measured wall-clock time in milliseconds. */
  totalMs: number;
  /** Measured completion tokens/sec over wall-clock time. */
  measured: number;
};

export type ModelAggregate = {
  model: string;
  vendor: ModelVendor;
  runs: number;
  decode: number;
  ttft: number;
  totalMs: number;
  measured: number;
};

export const DEFAULT_HISTORY_KEY = "browserai:v1:history";
export const DEFAULT_HISTORY_MAX_RECORDS = 200;

const VALID_VENDORS = new Set<ModelVendor>([
  "qwen",
  "google",
  "meta",
  "openai",
  "ibm",
  "microsoft",
  "nvidia",
  "liquid",
  "huggingface",
  "mistral",
  "other",
]);

export type RunHistoryOptions = {
  /** Persistence. Defaults to `globalThis.localStorage` when available; `null` = in-memory only. */
  storage?: StorageLike | null;
  key?: string;
  maxRecords?: number;
};

function defaultStorage(): StorageLike | null {
  try {
    // `localStorage` can throw in private-mode / sandboxed contexts; history becomes ephemeral.
    return (globalThis as { localStorage?: StorageLike }).localStorage ?? null;
  } catch {
    return null;
  }
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function coerceRecord(value: unknown): RunRecord | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const numericKeys = ["ts", "promptTokens", "completionTokens", "decode", "ttft", "totalMs", "measured"] as const;
  for (const key of numericKeys) {
    if (!isFiniteNumber(record[key])) return null;
  }
  if (typeof record.model !== "string" || typeof record.case !== "string" || record.case.length === 0) return null;

  return {
    ts: record.ts as number,
    model: record.model,
    vendor: VALID_VENDORS.has(record.vendor as ModelVendor) ? (record.vendor as ModelVendor) : "other",
    backend: record.backend === "transformers-js" ? "transformers-js" : "webllm",
    case: record.case,
    promptTokens: record.promptTokens as number,
    completionTokens: record.completionTokens as number,
    decode: record.decode as number,
    ttft: record.ttft as number,
    totalMs: record.totalMs as number,
    measured: record.measured as number,
  };
}

/**
 * A persisted, capped run log (newest first). All storage errors are swallowed: quota failures and
 * unavailable storage degrade to in-memory history rather than breaking generation flows.
 */
export class RunHistoryStore {
  readonly #storage: StorageLike | null;
  readonly #key: string;
  readonly #maxRecords: number;
  #records: RunRecord[];

  constructor(options: RunHistoryOptions = {}) {
    this.#storage = options.storage === undefined ? defaultStorage() : options.storage;
    this.#key = options.key ?? DEFAULT_HISTORY_KEY;
    this.#maxRecords = options.maxRecords ?? DEFAULT_HISTORY_MAX_RECORDS;
    this.#records = this.#load();
  }

  /** The in-memory records, newest first. */
  get records(): readonly RunRecord[] {
    return this.#records;
  }

  /** Prepend a record and persist. Returns the updated list. */
  append(record: RunRecord): readonly RunRecord[] {
    this.#records = [record, ...this.#records].slice(0, this.#maxRecords);
    this.#persist(this.#records);
    return this.#records;
  }

  /** Overwrite the history (e.g. after deleting selected rows). */
  replace(records: RunRecord[]): void {
    this.#records = records.slice(0, this.#maxRecords);
    this.#persist(this.#records);
  }

  clear(): void {
    this.#records = [];
    try {
      this.#storage?.removeItem(this.#key);
    } catch {
      // Clearing is best-effort.
    }
  }

  aggregateByModel(caseFilter: string | "all" = "all"): ModelAggregate[] {
    return aggregateByModel(this.#records, caseFilter);
  }

  #load(): RunRecord[] {
    if (!this.#storage) return [];
    try {
      const raw = this.#storage.getItem(this.#key);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.map(coerceRecord).filter((record): record is RunRecord => record !== null);
    } catch {
      return [];
    }
  }

  #persist(records: RunRecord[]): void {
    if (!this.#storage) return;
    try {
      this.#storage.setItem(this.#key, JSON.stringify(records));
    } catch {
      // Quota errors are non-fatal; the list still lives in memory for this session.
    }
  }
}

/**
 * Derive a vendor bucket from a preset. Uses explicit id/name matching rather than the repo owner
 * so re-hosted artifacts (e.g. onnx-community mirrors) still map to the true model vendor.
 */
export function vendorForModel(preset: ModelPreset): ModelVendor {
  const haystack = `${preset.id} ${preset.shortName}`.toLowerCase();
  if (haystack.includes("qwen")) return "qwen";
  if (haystack.includes("gemma")) return "google";
  if (haystack.includes("llama") || haystack.includes("hermes")) return "meta";
  if (haystack.includes("whisper")) return "openai";
  if (haystack.includes("granite")) return "ibm";
  if (haystack.includes("phi")) return "microsoft";
  if (haystack.includes("nemotron") || haystack.includes("parakeet")) return "nvidia";
  if (haystack.includes("lfm") || haystack.includes("liquid")) return "liquid";
  if (haystack.includes("smollm") || haystack.includes("smolvlm")) return "huggingface";
  if (haystack.includes("mistral") || haystack.includes("ministral") || haystack.includes("voxtral")) return "mistral";
  if (haystack.includes("olmo")) return "other";
  if (haystack.includes("minicpm")) return "other";
  return "other";
}

export const VENDOR_LABELS: Record<ModelVendor, string> = {
  qwen: "Qwen",
  google: "Google",
  meta: "Meta",
  openai: "OpenAI",
  ibm: "IBM",
  microsoft: "Microsoft",
  nvidia: "NVIDIA",
  liquid: "Liquid",
  huggingface: "HuggingFace",
  mistral: "Mistral",
  other: "Other",
};

/**
 * Group runs by model (optionally filtered by case) and average the per-run metrics.
 */
export function aggregateByModel(history: readonly RunRecord[], caseFilter: string | "all"): ModelAggregate[] {
  const buckets = new Map<string, { vendor: ModelVendor; runs: number; decode: number; ttft: number; totalMs: number; measured: number }>();

  for (const record of history) {
    if (caseFilter !== "all" && record.case !== caseFilter) continue;
    const bucket = buckets.get(record.model) ?? { vendor: record.vendor, runs: 0, decode: 0, ttft: 0, totalMs: 0, measured: 0 };
    bucket.runs += 1;
    bucket.decode += record.decode;
    bucket.ttft += record.ttft;
    bucket.totalMs += record.totalMs;
    bucket.measured += record.measured;
    buckets.set(record.model, bucket);
  }

  return Array.from(buckets.entries()).map(([model, bucket]) => ({
    model,
    vendor: bucket.vendor,
    runs: bucket.runs,
    decode: bucket.decode / bucket.runs,
    ttft: bucket.ttft / bucket.runs,
    totalMs: bucket.totalMs / bucket.runs,
    measured: bucket.measured / bucket.runs,
  }));
}
