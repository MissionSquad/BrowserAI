/// <reference path="./runtime-types.d.ts" />
import * as ort from 'onnxruntime-tdt/webgpu';
import { phononFeatures } from './frontend.js';
import { nemoFeatures } from './nemo-frontend.js';

export interface TdtSegment { start: number; end: number; text: string }
export interface TdtModel {
  transcribe(audio: Float32Array, onDelta?: (text: string) => void, signal?: AbortSignal): Promise<{ text: string; segments: TdtSegment[] }>;
  dispose(): Promise<void>;
}
export interface TdtLoadOptions {
  device: 'webgpu' | 'wasm'; baseUrl?: string; signal?: AbortSignal;
  wasmPaths?: string | { mjs: string; wasm: string };
  onProgress?: (info: { status?: string; progress?: number; file?: string }) => void;
}
export function checkAbort(signal?: AbortSignal): void { signal?.throwIfAborted(); }
/** Let queued input/cancellation events run even when WASM resolves only microtasks. */
async function yieldToBrowser(signal?: AbortSignal): Promise<void> {
  checkAbort(signal);
  await new Promise<void>(resolve => setTimeout(resolve, 0));
  checkAbort(signal);
}
export function parseVocabulary(text: string): string[] {
  const vocab: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line) continue;
    const match = /^(.*) (\d+)$/.exec(line);
    if (!match) throw new Error('Invalid TDT vocabulary line.');
    const id = Number(match[2]);
    if (vocab[id] !== undefined) throw new Error('Duplicate TDT vocabulary ID.');
    vocab[id] = match[1];
  }
  if (!vocab.length || Array.from({ length: vocab.length }, (_, i) => vocab[i]).some(x => x === undefined)) throw new Error('Incomplete TDT vocabulary.');
  return vocab;
}
export function decodePieces(ids: number[], vocab: string[]): string {
  return ids.map(id => vocab[id] ?? '').filter(piece => piece !== '<unk>' && piece !== '<pad>' && piece !== '<blk>' && !/^<\|.*\|>$/.test(piece)).join('').replace(/▁/g, ' ').trim();
}
/** Split near the quietest 100 ms interval in the final five seconds, bounded at 30 seconds. */
export function splitAudio(audio: Float32Array): Array<{ start: number; end: number }> {
  const chunks: Array<{ start: number; end: number }> = [];
  let start = 0;
  while (audio.length - start > 480000) {
    let end = start + 480000, lowest = Infinity;
    for (let p = start + 400000; p <= start + 478400; p += 1600) {
      let energy = 0;
      for (let i = p; i < p + 1600; i++) energy += audio[i] ** 2;
      if (energy < lowest) { lowest = energy; end = p + 800; }
    }
    chunks.push({ start, end }); start = end;
  }
  if (start < audio.length) chunks.push({ start, end: audio.length });
  return chunks;
}
async function artifact(url: string, options: TdtLoadOptions): Promise<Uint8Array> {
  checkAbort(options.signal);
  const cacheWarning = (): void => options.onProgress?.({ status: 'cache-warning', file: url });
  let cache: Cache | undefined;
  let response: Response | undefined;
  try { cache = typeof caches !== 'undefined' ? await caches.open('transformers-cache') : undefined; }
  catch { cacheWarning(); }
  try { response = await cache?.match(url); }
  catch { cacheWarning(); }
  const cachePut = async (value: Response): Promise<void> => {
    if (!cache) return;
    try { await cache.put(url, value); }
    catch { cacheWarning(); }
  };
  if (!response) {
    response = await fetch(url, { signal: options.signal });
    if (!response.ok) throw new Error(`TDT download failed (${response.status}): ${url}`);
    // Consume incrementally for meaningful download progress, then cache only a complete artifact.
    const total = Number(response.headers.get('content-length'));
    const reader = response.body?.getReader();
    if (reader) {
      const chunks: Uint8Array[] = []; let received = 0;
      try {
        while (true) {
          checkAbort(options.signal);
          const next = await reader.read(); if (next.done) break;
          chunks.push(next.value); received += next.value.length;
          options.onProgress?.({ status: 'progress', file: url, ...(total > 0 ? { progress: Math.min(100, received / total * 100) } : {}) });
        }
      } finally { reader.releaseLock(); }
      const bytes = new Uint8Array(received); let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
      checkAbort(options.signal);
      await cachePut(new Response(bytes, { headers: response.headers }));
      options.onProgress?.({ status: 'done', file: url, progress: 100 });
      return bytes;
    }
    await cachePut(response.clone());
  }
  if (!response.ok) throw new Error(`Invalid cached TDT artifact (${response.status}): ${url}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  checkAbort(options.signal);
  options.onProgress?.({ status: 'done', file: url, progress: 100 });
  return bytes;
}
function floats(tensor: ort.Tensor | undefined, name: string): Float32Array {
  if (!tensor || tensor.type !== 'float32' || !(tensor.data instanceof Float32Array)) throw new Error(`Invalid TDT ${name} tensor.`);
  return tensor.data;
}
function releaseTensors(values: Record<string, ort.Tensor>): void { for (const tensor of new Set(Object.values(values))) tensor.dispose(); }
function argmax(values: Float32Array, start: number, end: number): number {
  let best = start; for (let i = start + 1; i < end; i++) if (values[i] > values[best]) best = i;
  return best - start;
}

/** Load the two explicitly supported ONNX TDT exports. Sessions are released on any partial load failure. */
export async function loadTdtModel(modelId: string, options: TdtLoadOptions): Promise<TdtModel> {
  const phonon = modelId === 'enriqueag/phonon-2.5-onnx';
  if (!phonon && modelId !== 'istupakov/parakeet-tdt-0.6b-v3-onnx') throw new Error(`Unsupported TDT model: ${modelId}`);
  if (phonon && options.device !== 'webgpu') throw new Error('Phonon exact2x2 requires WebGPU.');
  const base = options.baseUrl ?? `https://huggingface.co/${modelId}/resolve/main/`;
  const url = (file: string): string => new URL(file, base.endsWith('/') ? base : `${base}/`).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = options.wasmPaths ?? {
    mjs: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort-wasm-simd-threaded.asyncify.mjs',
    wasm: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort-wasm-simd-threaded.asyncify.wasm',
  };
  const sessions: ort.InferenceSession[] = [];
  const create = async (file: string, device: 'webgpu' | 'wasm', decoder = false): Promise<ort.InferenceSession> => {
    const bytes = await artifact(url(file), options); checkAbort(options.signal);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: [device], ...(decoder && phonon ? { extra: { session: { disable_quant_qdq: '1' } } } : {}) });
    sessions.push(session); checkAbort(options.signal); return session;
  };
  try {
    const vocab = parseVocabulary(new TextDecoder().decode(await artifact(url('vocab.txt'), options)));
    const blank = phonon ? 8192 : vocab.indexOf('<blk>');
    if (blank < 0 || (phonon && vocab.length !== 8192 && vocab.length !== 8193)) throw new Error('Invalid TDT blank/vocabulary configuration.');
    const encoder = await create(phonon ? 'encoder.exact2x2.onnx' : 'encoder-model.int8.onnx', phonon ? 'webgpu' : 'wasm');
    const decoder = await create(phonon ? 'decoder_joint.int8.onnx' : 'decoder_joint-model.int8.onnx', 'wasm', true);
    let disposed = false, busy = false;
    const stateShape = [2, 1, 640];
    return {
      async dispose() {
        if (busy) throw new Error('Cannot dispose TDT model during transcription.');
        if (!disposed) { disposed = true; await Promise.all(sessions.map(s => s.release())); }
      },
      async transcribe(audio, onDelta, signal) {
        if (disposed) throw new Error('TDT model is disposed.');
        if (busy) throw new Error('TDT model is already transcribing.');
        if (!(audio instanceof Float32Array) || audio.some(x => !Number.isFinite(x))) throw new Error('TDT requires finite mono 16 kHz Float32 audio.');
        checkAbort(signal); busy = true;
        const segments: TdtSegment[] = [];
        let runsSinceYield = 0, lastYield = performance.now();
        try {
          for (const chunk of splitAudio(audio)) {
            checkAbort(signal);
            const wave = audio.slice(chunk.start, chunk.end);
            if (wave.length < 320) continue;
            const resources = new Set<ort.Tensor>();
            const own = <T extends ort.Tensor>(t: T): T => { resources.add(t); return t; };
            const run = async (session: ort.InferenceSession, feeds: Record<string, ort.Tensor>): Promise<Record<string, ort.Tensor>> => {
              checkAbort(signal);
              if (++runsSinceYield >= 32 || performance.now() - lastYield >= 20) {
                await yieldToBrowser(signal); runsSinceYield = 0; lastYield = performance.now();
              }
              const outputs = await session.run(feeds);
              for (const tensor of Object.values(outputs)) own(tensor);
              checkAbort(signal); return outputs;
            };
            try {
              let encoded: Record<string, ort.Tensor>;
              if (phonon) {
                await yieldToBrowser(signal);
                const features = phononFeatures(wave);
                await yieldToBrowser(signal);
                runsSinceYield = 0; lastYield = performance.now();
                encoded = await run(encoder, { features: own(new ort.Tensor('float32', features.data, [1, features.frames, 128])) });
              } else {
                // The published nemo128 graph uses float64 STFT, unsupported by ORT Web.
                // This front end reproduces that graph's reflect padding and statistics.
                await yieldToBrowser(signal);
                const features = nemoFeatures(wave);
                await yieldToBrowser(signal);
                runsSinceYield = 0; lastYield = performance.now();
                encoded = await run(encoder, {
                  audio_signal: own(new ort.Tensor('float32', features.data, [1, 128, features.frames])),
                  length: own(new ort.Tensor('int64', BigInt64Array.of(BigInt(features.frames)), [1])),
                });
              }
              // Large encoder runs are synchronous on WASM. Give input handlers a turn
              // before entering the otherwise microtask-only decoder loop.
              await yieldToBrowser(signal); runsSinceYield = 0; lastYield = performance.now();
              const output = encoded[phonon ? 'encoded' : 'outputs'];
              const data = floats(output, 'encoder');
              const width = output.dims[phonon ? 2 : 1], paddedSteps = output.dims[phonon ? 1 : 2];
              const lengths = encoded.encoded_lengths;
              const steps = phonon ? paddedSteps : Number(lengths?.data[0]);
              if (!Number.isInteger(steps) || steps < 0 || steps > paddedSteps) throw new Error('Invalid TDT encoder length.');
              let h: ort.Tensor = own(new ort.Tensor('float32', new Float32Array(stateShape.reduce((a, b) => a * b, 1)), stateShape));
              let c: ort.Tensor = own(new ort.Tensor('float32', new Float32Array(stateShape.reduce((a, b) => a * b, 1)), stateShape));
              let token = blank, t = 0, emitted = 0;
              const ids: number[] = [];
              while (t < steps) {
                const frame = new Float32Array(width);
                for (let i = 0; i < width; i++) frame[i] = data[phonon ? t * width + i : i * paddedSteps + t];
                const ephemeral: Record<string, ort.Tensor> = phonon ? {
                  encoded: new ort.Tensor('float32', frame, [1, width]), token: new ort.Tensor('int32', Int32Array.of(token), [1]),
                } : {
                  encoder_outputs: new ort.Tensor('float32', frame, [1, width, 1]), targets: new ort.Tensor('int32', Int32Array.of(token), [1, 1]), target_length: new ort.Tensor('int32', Int32Array.of(1), [1]),
                };
                let result: Record<string, ort.Tensor>;
                try { result = await run(decoder, { ...ephemeral, ...(phonon ? { h, c } : { input_states_1: h, input_states_2: c }) }); }
                finally { releaseTensors(ephemeral); }
                const logits = floats(result[phonon ? 'logits' : 'outputs'], 'decoder');
                if (logits.length !== blank + 1 + 5) throw new Error('Invalid TDT token/duration logits.');
                const next = argmax(logits, 0, blank + 1); let duration = argmax(logits, blank + 1, logits.length);
                if (next !== blank) {
                  ids.push(next); token = next; emitted++;
                  resources.delete(h); h.dispose(); resources.delete(c); c.dispose();
                  h = result[phonon ? 'h_out' : 'output_states_1']; c = result[phonon ? 'c_out' : 'output_states_2'];
                }
                for (const tensor of Object.values(result)) if (tensor !== h && tensor !== c) { resources.delete(tensor); tensor.dispose(); }
                if ((next === blank || emitted >= 10) && duration === 0) duration = 1;
                if (duration > 0) emitted = 0;
                t += duration;
              }
              const text = decodePieces(ids, vocab);
              if (text) { segments.push({ start: chunk.start / 16000, end: chunk.end / 16000, text }); onDelta?.(segments.map(segment => segment.text).join(' ')); }
            } finally { for (const tensor of resources) tensor.dispose(); }
          }
          return { text: segments.map(s => s.text).join(' '), segments };
        } finally { busy = false; }
      },
    };
  } catch (error) { await Promise.allSettled(sessions.map(s => s.release())); throw error; }
}
