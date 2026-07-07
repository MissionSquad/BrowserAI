import { describe, expect, it } from "vitest";
import type { StorageLike } from "../src/config.js";
import type { RunRecord } from "../src/history.js";
import { DEFAULT_HISTORY_KEY, RunHistoryStore } from "../src/history.js";
import { runRecordFromStats, wordCountGenStats } from "../src/metrics.js";
import type { RuntimeStats } from "../src/inference.js";

function makeStorage(initial: Record<string, string> = {}): StorageLike & { data: Map<string, string> } {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => void data.set(key, value),
    removeItem: (key) => void data.delete(key),
  };
}

function record(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    ts: 1,
    model: "Qwen3.5 0.8B",
    vendor: "qwen",
    backend: "webllm",
    case: "chat",
    promptTokens: 10,
    completionTokens: 20,
    decode: 30,
    ttft: 0.5,
    totalMs: 1000,
    measured: 20,
    ...overrides,
  };
}

describe("RunHistoryStore", () => {
  it("persists appended records newest-first under the default key", () => {
    const storage = makeStorage();
    const store = new RunHistoryStore({ storage });
    store.append(record({ ts: 1 }));
    store.append(record({ ts: 2 }));
    expect(store.records.map((r) => r.ts)).toEqual([2, 1]);
    const persisted = JSON.parse(storage.data.get(DEFAULT_HISTORY_KEY)!);
    expect(persisted).toHaveLength(2);
    expect(persisted[0].ts).toBe(2);
  });

  it("caps the history at maxRecords", () => {
    const store = new RunHistoryStore({ storage: null, maxRecords: 3 });
    for (let i = 0; i < 5; i += 1) store.append(record({ ts: i }));
    expect(store.records).toHaveLength(3);
    expect(store.records[0].ts).toBe(4);
  });

  it("loads existing records and drops malformed rows silently", () => {
    const storage = makeStorage({
      [DEFAULT_HISTORY_KEY]: JSON.stringify([record({ ts: 7 }), { junk: true }, "nope", record({ ts: 8, case: "" })]),
    });
    const store = new RunHistoryStore({ storage });
    expect(store.records.map((r) => r.ts)).toEqual([7]);
  });

  it("coerces unknown vendors/backends and accepts custom case strings", () => {
    const storage = makeStorage({
      [DEFAULT_HISTORY_KEY]: JSON.stringify([record({ vendor: "acme" as never, backend: "vllm" as never, case: "my-task" })]),
    });
    const store = new RunHistoryStore({ storage });
    expect(store.records[0].vendor).toBe("other");
    expect(store.records[0].backend).toBe("webllm");
    expect(store.records[0].case).toBe("my-task");
  });

  it("survives a throwing storage (history becomes in-memory only)", () => {
    const storage: StorageLike = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("quota");
      },
      removeItem: () => {
        throw new Error("blocked");
      },
    };
    const store = new RunHistoryStore({ storage });
    store.append(record());
    expect(store.records).toHaveLength(1);
    store.clear();
    expect(store.records).toHaveLength(0);
  });

  it("aggregates per model with case filtering", () => {
    const store = new RunHistoryStore({ storage: null });
    store.append(record({ model: "A", decode: 10, case: "chat" }));
    store.append(record({ model: "A", decode: 20, case: "chat" }));
    store.append(record({ model: "A", decode: 99, case: "json" }));
    const chat = store.aggregateByModel("chat");
    expect(chat).toHaveLength(1);
    expect(chat[0].runs).toBe(2);
    expect(chat[0].decode).toBe(15);
    expect(store.aggregateByModel("all")[0].runs).toBe(3);
  });
});

describe("runRecordFromStats", () => {
  const stats: RuntimeStats = {
    rawText: "",
    model: "Qwen3.5-0.8B-q4f16_1-MLC",
    backend: "webllm",
    finishReason: "stop",
    measuredElapsedMs: 1200,
    promptTokens: 15,
    completionTokens: 42,
    measuredCompletionTokensPerSecond: 35,
    extra: { decode_tokens_per_s: 40, time_to_first_token_s: 0.3 },
  };

  it("maps stats to a record attributed to the producing model's preset", () => {
    const rec = runRecordFromStats("chat", stats);
    expect(rec).not.toBeNull();
    expect(rec!.model).toBe("Qwen3.5 0.8B");
    expect(rec!.vendor).toBe("qwen");
    expect(rec!.decode).toBe(40);
    expect(rec!.ttft).toBe(0.3);
    expect(rec!.measured).toBe(35);
  });

  it("falls back to measured decode when engine telemetry is absent", () => {
    const rec = runRecordFromStats("chat", { ...stats, extra: undefined });
    expect(rec!.decode).toBe(35);
    expect(rec!.ttft).toBe(0);
  });

  it("returns null for stats without a model (error placeholders never pollute history)", () => {
    expect(runRecordFromStats("chat", { ...stats, model: "n/a" })).toBeNull();
    expect(runRecordFromStats("chat", { ...stats, model: "" })).toBeNull();
  });
});

describe("wordCountGenStats", () => {
  it("counts words as a token proxy for non-generative ASR", () => {
    expect(wordCountGenStats("  hello there   world ", 0, 100).completionTokens).toBe(3);
    expect(wordCountGenStats("", 0, 100).completionTokens).toBe(0);
  });
});
