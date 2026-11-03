/** Phonon reference front end: NeMo log-mel, without dither. */
export function phononFeatures(audio: Float32Array): { data: Float32Array; frames: number } {
  const frames = Math.floor(audio.length / 160);
  if (frames < 2) throw new Error('TDT audio must contain at least 320 samples (20 ms).');
  const emphasized = new Float32Array(audio.length);
  emphasized[0] = audio[0];
  for (let i = 1; i < audio.length; i++) emphasized[i] = audio[i] - .97 * audio[i - 1];
  const hzToMel = (hz: number): number => hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27);
  const melToHz = (mel: number): number => mel < 15 ? mel * (200 / 3) : 1000 * Math.exp((mel - 15) * Math.log(6.4) / 27);
  const edges = Array.from({ length: 130 }, (_, i) => melToHz(hzToMel(8000) * i / 129));
  const filters = Array.from({ length: 128 }, (_, m) => Float64Array.from({ length: 257 }, (_, k) => {
    const hz = k * 16000 / 512;
    return Math.max(0, Math.min((hz - edges[m]) / (edges[m + 1] - edges[m]), (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]))) * 2 / (edges[m + 2] - edges[m]);
  }));
  const window = Float64Array.from({ length: 400 }, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * i / 399));
  const data = new Float32Array(frames * 128);
  const real = new Float64Array(512), imag = new Float64Array(512);
  for (let f = 0; f < frames; f++) {
    real.fill(0); imag.fill(0);
    for (let i = 0; i < 400; i++) {
      const sample = f * 160 + i - 200;
      real[i + 56] = (sample >= 0 && sample < audio.length ? emphasized[sample] : 0) * window[i];
    }
    fft(real, imag);
    for (let m = 0; m < 128; m++) {
      let power = 0;
      for (let k = 0; k <= 256; k++) power += (real[k] ** 2 + imag[k] ** 2) * filters[m][k];
      data[f * 128 + m] = Math.log(power + 2 ** -24);
    }
  }
  for (let m = 0; m < 128; m++) {
    let mean = 0;
    for (let f = 0; f < frames; f++) mean += data[f * 128 + m];
    mean /= frames;
    let variance = 0;
    for (let f = 0; f < frames; f++) variance += (data[f * 128 + m] - mean) ** 2;
    const std = Math.sqrt(variance / (frames - 1)) + 1e-5;
    for (let f = 0; f < frames; f++) data[f * 128 + m] = (data[f * 128 + m] - mean) / std;
  }
  return { data, frames };
}

export function fft(real: Float64Array, imag: Float64Array): void {
  for (let i = 1, j = 0; i < 512; i++) {
    let bit = 256;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [real[i], real[j]] = [real[j], real[i]]; }
  }
  for (let size = 2; size <= 512; size *= 2) {
    for (let start = 0; start < 512; start += size) {
      for (let j = 0; j < size / 2; j++) {
        const angle = -2 * Math.PI * j / size, c = Math.cos(angle), s = Math.sin(angle);
        const a = start + j, b = a + size / 2;
        const r = c * real[b] - s * imag[b], im = s * real[b] + c * imag[b];
        real[b] = real[a] - r; imag[b] = imag[a] - im;
        real[a] += r; imag[a] += im;
      }
    }
  }
}
