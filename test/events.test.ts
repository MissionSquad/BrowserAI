import { describe, expect, it, vi } from "vitest";
import { TypedEmitter } from "../src/events.js";

type Events = { ping: { n: number }; empty: Record<string, never> };

class Emitter extends TypedEmitter<Events> {
  fire<K extends keyof Events>(event: K, payload: Events[K]): void {
    this.emit(event, payload);
  }
}

describe("TypedEmitter", () => {
  it("delivers payloads to subscribers and honors unsubscribe", () => {
    const emitter = new Emitter();
    const seen: number[] = [];
    const off = emitter.on("ping", ({ n }) => seen.push(n));
    emitter.fire("ping", { n: 1 });
    off();
    emitter.fire("ping", { n: 2 });
    expect(seen).toEqual([1]);
  });

  it("once() fires exactly once", () => {
    const emitter = new Emitter();
    const listener = vi.fn();
    emitter.once("ping", listener);
    emitter.fire("ping", { n: 1 });
    emitter.fire("ping", { n: 2 });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("a throwing listener does not starve later listeners", () => {
    const emitter = new Emitter();
    const seen: number[] = [];
    emitter.on("ping", () => {
      throw new Error("boom");
    });
    emitter.on("ping", ({ n }) => seen.push(n));
    emitter.fire("ping", { n: 3 });
    expect(seen).toEqual([3]);
  });

  it("a listener unsubscribing mid-emit does not skip other deliveries", () => {
    const emitter = new Emitter();
    const seen: string[] = [];
    const offA = emitter.on("empty", () => {
      seen.push("a");
      offA();
    });
    emitter.on("empty", () => seen.push("b"));
    emitter.fire("empty", {});
    emitter.fire("empty", {});
    expect(seen).toEqual(["a", "b", "b"]);
  });
});
