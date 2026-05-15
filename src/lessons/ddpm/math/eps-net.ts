/**
 * Browser-side forward pass of the trained epsilon network.
 *
 * Architecture (matching the offline training notebook):
 *   input  = concat(x_t in R^2, sinusoidal_time_embed(t) in R^{time_dim})
 *   net    = Linear -> SiLU -> Linear -> SiLU -> Linear -> SiLU -> Linear
 *   output = eps_theta in R^2
 *
 * Time embedding (Vaswani-style):
 *   half = time_dim // 2
 *   freqs[k] = exp(-log(10000) * k / (half - 1)),  k = 0..half-1
 *   emb = concat(sin(t * freqs), cos(t * freqs))
 *
 * All linear layers are stored as [out, in] row-major arrays.
 */

export interface DDPMWeights {
  'net.0.weight': number[][]; 'net.0.bias': number[];
  'net.2.weight': number[][]; 'net.2.bias': number[];
  'net.4.weight': number[][]; 'net.4.bias': number[];
  'net.6.weight': number[][]; 'net.6.bias': number[];
  _metadata: {
    T: number;
    betas: number[];
    alpha_bars: number[];
    data_centers: number[][];
    hidden_dim: number;
    time_dim: number;
    epochs: number;
    schedule?: string;
  };
}

function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

function dense(input: number[], W: number[][], b: number[]): number[] {
  const out = new Array<number>(W.length);
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    let s = b[i];
    for (let j = 0; j < row.length; j++) s += row[j] * input[j];
    out[i] = s;
  }
  return out;
}

/** Sinusoidal time embedding. t is an integer timestep. */
export function timeEmbed(t: number, time_dim: number): number[] {
  const half = Math.floor(time_dim / 2);
  const emb = new Array<number>(time_dim);
  for (let k = 0; k < half; k++) {
    const freq = Math.exp(-Math.log(10000) * k / (half - 1));
    const phase = t * freq;
    emb[k] = Math.sin(phase);
    emb[k + half] = Math.cos(phase);
  }
  return emb;
}

/** Forward pass: eps_theta(x_t, t). */
export function epsNetForward(x_t: number[], t: number, w: DDPMWeights): number[] {
  const td = w._metadata.time_dim;
  const input = [...x_t, ...timeEmbed(t, td)];
  const h1 = dense(input, w['net.0.weight'], w['net.0.bias']).map(silu);
  const h2 = dense(h1,    w['net.2.weight'], w['net.2.bias']).map(silu);
  const h3 = dense(h2,    w['net.4.weight'], w['net.4.bias']).map(silu);
  return       dense(h3,    w['net.6.weight'], w['net.6.bias']);
}

/**
 * Batched forward pass for a list of (x, t) inputs.
 * When all entries share the same t, the time embedding is computed once.
 */
export function epsNetForwardBatch(xs: number[][], ts: number[], w: DDPMWeights): number[][] {
  return xs.map((x, i) => epsNetForward(x, ts[i], w));
}

export async function loadDDPMWeights(url: string): Promise<DDPMWeights | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json() as DDPMWeights;
  } catch {
    return null;
  }
}
