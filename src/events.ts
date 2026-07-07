/**
 * Minimal typed event emitter. Framework-agnostic and dependency-free; the event map type gives
 * hosts full payload typing (`on("stagechange", ({ stage }) => …)`).
 */
export type EventMap = Record<string, unknown>;

export type Listener<T> = (payload: T) => void;

export class TypedEmitter<Events extends EventMap> {
  #listeners = new Map<keyof Events, Set<Listener<never>>>();

  /** Subscribe. Returns an unsubscribe function. */
  on<K extends keyof Events>(event: K, listener: Listener<Events[K]>): () => void {
    let set = this.#listeners.get(event);
    if (!set) {
      set = new Set();
      this.#listeners.set(event, set);
    }
    set.add(listener as Listener<never>);
    return () => this.off(event, listener);
  }

  /** Subscribe for a single emission. Returns an unsubscribe function. */
  once<K extends keyof Events>(event: K, listener: Listener<Events[K]>): () => void {
    const unsubscribe = this.on(event, (payload) => {
      unsubscribe();
      listener(payload);
    });
    return unsubscribe;
  }

  off<K extends keyof Events>(event: K, listener: Listener<Events[K]>): void {
    this.#listeners.get(event)?.delete(listener as Listener<never>);
  }

  protected emit<K extends keyof Events>(event: K, payload: Events[K]): void {
    const set = this.#listeners.get(event);
    if (!set) return;
    // Copy so a listener unsubscribing (or subscribing) mid-emit cannot skip/duplicate deliveries.
    for (const listener of [...set]) {
      try {
        (listener as Listener<Events[K]>)(payload);
      } catch {
        // A throwing listener must not break the engine or starve other listeners.
      }
    }
  }

  /** Remove every listener (all events). */
  removeAllListeners(): void {
    this.#listeners.clear();
  }
}
