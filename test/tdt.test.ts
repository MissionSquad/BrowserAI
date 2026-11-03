import { describe, it, expect, vi, beforeEach } from 'vitest';
const mocks = vi.hoisted(() => ({ create: vi.fn(), sessions: [] as Array<{ run: ReturnType<typeof vi.fn>; release: ReturnType<typeof vi.fn> }> }));
vi.mock('onnxruntime-tdt/webgpu', () => ({
  env: { wasm: {} },
  Tensor: class {
    disposed = false;
    constructor(public type: string, public data: Float32Array | Int32Array | BigInt64Array, public dims: number[]) {}
    dispose() { this.disposed = true; }
  },
  InferenceSession: { create: mocks.create },
}));
import { loadTdtModel, parseVocabulary, decodePieces, splitAudio } from '../src/tdt/index.js';
import { Tensor } from 'onnxruntime-tdt/webgpu';
const tensor = (data: Float32Array, dims: number[]) => new Tensor('float32', data, dims);
const vocab = Array.from({ length: 8192 }, (_, i) => `${i === 1 ? '▁Hello' : i === 2 ? '▁world' : '<unk>'} ${i}`).join('\n');
beforeEach(() => {
  vi.restoreAllMocks(); mocks.create.mockReset(); mocks.sessions.length = 0;
  vi.stubGlobal('caches', undefined);
  vi.stubGlobal('fetch', vi.fn(async (url: string) => new Response(url.endsWith('vocab.txt') ? vocab : new Uint8Array([1]))));
});
function session(run: ReturnType<typeof vi.fn>) { const s = { run, release: vi.fn(async () => {}) }; mocks.sessions.push(s); return s; }
function logits(token: number, duration: number) {
  const data = new Float32Array(8198).fill(-10); data[token] = 10; data[8193 + duration] = 10;
  return { logits: tensor(data, [1, 8198]), h_out: tensor(new Float32Array(1280), [2, 1, 640]), c_out: tensor(new Float32Array(1280), [2, 1, 640]) };
}
describe('TDT adapter', () => {
  it('preserves state only for nonblank tokens and advances blank zero-duration frames', async () => {
    const first = logits(1, 0), blank = logits(8192, 0), last = logits(2, 1);
    const encoder = session(vi.fn(async () => ({ encoded: tensor(new Float32Array(1280), [1, 2, 640]) })));
    const decoder = session(vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(blank).mockResolvedValueOnce(last));
    mocks.create.mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder);
    const model = await loadTdtModel('enriqueag/phonon-2.5-onnx', { device: 'webgpu' });
    const result = await model.transcribe(new Float32Array(1600));
    expect(result.text).toBe('Hello world'); expect(result.segments).toEqual([{ start: 0, end: .1, text: 'Hello world' }]);
    expect(decoder.run.mock.calls[1][0].h).toBe(first.h_out);
    expect(decoder.run.mock.calls[2][0].h).toBe(first.h_out);
    expect(decoder.run.mock.calls[2][0].token.data[0]).toBe(1);
    await model.dispose(); expect(encoder.release).toHaveBeenCalledOnce(); expect(decoder.release).toHaveBeenCalledOnce();
  });
  it('caps duration-zero emission at ten tokens and checks abort between runs', async () => {
    const control = new AbortController();
    const encoder = session(vi.fn(async () => ({ encoded: tensor(new Float32Array(640), [1, 1, 640]) })));
    const decoder = session(vi.fn(async () => logits(1, 0)));
    mocks.create.mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder);
    const model = await loadTdtModel('enriqueag/phonon-2.5-onnx', { device: 'webgpu' });
    await model.transcribe(new Float32Array(1600)); expect(decoder.run).toHaveBeenCalledTimes(10);
    decoder.run.mockImplementation(async () => { control.abort(); return logits(1, 1); });
    await expect(model.transcribe(new Float32Array(1600), undefined, control.signal)).rejects.toMatchObject({ name: 'AbortError' });
    await model.dispose();
  });
  it('yields to queued browser cancellation events during microtask-only decoding', async () => {
    const control = new AbortController();
    const encoder = session(vi.fn(async () => ({ encoded: tensor(new Float32Array(64000), [1, 100, 640]) })));
    const decoder = session(vi.fn(async () => {
      if (decoder.run.mock.calls.length === 1) setTimeout(() => control.abort(), 0);
      return logits(8192, 1);
    }));
    mocks.create.mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder);
    const model = await loadTdtModel('enriqueag/phonon-2.5-onnx', { device: 'webgpu' });
    await expect(model.transcribe(new Float32Array(1600), undefined, control.signal)).rejects.toMatchObject({ name: 'AbortError' });
    expect(decoder.run.mock.calls.length).toBeGreaterThan(0);
    expect(decoder.run.mock.calls.length).toBeLessThanOrEqual(32);
    await model.dispose();
  });
  it('uses browser-safe NeMo features and the int8 WASM encoder contract', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => new Response(url.endsWith('vocab.txt') ? `${vocab}\n<blk> 8192` : new Uint8Array([1]))));
    const encoder = session(vi.fn(async () => ({ outputs: tensor(new Float32Array(1024), [1, 1024, 1]), encoded_lengths: new Tensor('int64', BigInt64Array.of(1n), [1]) })));
    const decoded = logits(1, 1);
    const decoder = session(vi.fn(async () => ({ outputs: decoded.logits, output_states_1: decoded.h_out, output_states_2: decoded.c_out })));
    mocks.create.mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder);
    const model = await loadTdtModel('istupakov/parakeet-tdt-0.6b-v3-onnx', { device: 'webgpu' });
    expect((await model.transcribe(new Float32Array(1600))).text).toBe('Hello');
    expect(encoder.run.mock.calls[0][0].audio_signal.dims).toEqual([1, 128, 11]);
    expect(encoder.run.mock.calls[0][0].length.data[0]).toBe(11n);
    expect(fetch).toHaveBeenCalledTimes(3);
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('nemo128.onnx'))).toBe(false);
    expect(decoder.run.mock.calls[0][0].encoder_outputs.dims).toEqual([1, 1024, 1]);
    expect(decoder.run.mock.calls[0][0].targets.type).toBe('int32');
    for (const call of mocks.create.mock.calls) expect(call[1].executionProviders).toEqual(['wasm']);
    await model.dispose();
  });
  it('releases completed sessions when a later session fails', async () => {
    const encoder = session(vi.fn());
    mocks.create.mockResolvedValueOnce(encoder).mockRejectedValueOnce(new Error('decoder failure'));
    await expect(loadTdtModel('enriqueag/phonon-2.5-onnx', { device: 'webgpu' })).rejects.toThrow('decoder failure');
    expect(encoder.release).toHaveBeenCalledOnce();
  });
  it.each(['open', 'match', 'put'])('tolerates cache %s failures and continues inference', async failure => {
    const cache = {
      match: vi.fn(async () => { if (failure === 'match') throw new Error('cache read failed'); return undefined; }),
      put: vi.fn(async () => { if (failure === 'put') throw new Error('cache write failed'); }),
    };
    vi.stubGlobal('caches', { open: vi.fn(async () => { if (failure === 'open') throw new Error('cache unavailable'); return cache; }) });
    const encoder = session(vi.fn(async () => ({ encoded: tensor(new Float32Array(640), [1, 1, 640]) })));
    const decoder = session(vi.fn(async () => logits(1, 1)));
    mocks.create.mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder);
    const progress = vi.fn();
    const model = await loadTdtModel('enriqueag/phonon-2.5-onnx', { device: 'webgpu', onProgress: progress });
    expect((await model.transcribe(new Float32Array(1600))).text).toBe('Hello');
    expect(fetch).toHaveBeenCalledTimes(3);
    expect(progress.mock.calls.some(([info]) => info.status === 'cache-warning')).toBe(true);
    await model.dispose();
  });
  it('rejects HTTP failures before session creation', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('missing', { status: 404 })));
    await expect(loadTdtModel('enriqueag/phonon-2.5-onnx', { device: 'webgpu' })).rejects.toThrow('404'); expect(mocks.create).not.toHaveBeenCalled();
  });
  it('splits long audio without gaps or loss and bounds every chunk', () => {
    const audio = new Float32Array(16000 * 71); const chunks = splitAudio(audio);
    expect(chunks[0].start).toBe(0); expect(chunks.at(-1)?.end).toBe(audio.length);
    chunks.forEach((chunk, i) => { expect(chunk.end - chunk.start).toBeLessThanOrEqual(480000); if (i) expect(chunk.start).toBe(chunks[i - 1].end); });
  });
  it('parses pieces without losing spaces and filters control tokens', () => {
    expect(decodePieces([0, 1, 2, 3], parseVocabulary('▁Hi 0\n▁there 1\n<|en|> 2\n<unk> 3'))).toBe('Hi there');
    expect(() => parseVocabulary('a 1')).toThrow('Incomplete');
  });
});
