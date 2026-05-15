export interface ScoreNetWeights {
  'net.0.weight': number[][];   // hidden × 3
  'net.0.bias':   number[];     // hidden
  'net.2.weight': number[][];   // hidden × hidden
  'net.2.bias':   number[];     // hidden
  'net.4.weight': number[][];   // 2 × hidden
  'net.4.bias':   number[];     // 2
  _metadata: {
    sigmas: number[];
    data_centers: number[][];
    hidden_dim: number;
    epochs: number;
  };
}

function tanh(x: number): number {
  return Math.tanh(x);
}

function linearLayer(input: number[], W: number[][], b: number[]): number[] {
  return W.map((row, i) => b[i] + row.reduce((s, w, j) => s + w * input[j], 0));
}

/**
 * Forward pass of the score network: s_theta(x, sigma) → score in R^2.
 * Input: concat(x, log(sigma)) — 3 dimensions.
 */
export function scoreNetForward(x: number[], sigma: number, weights: ScoreNetWeights): number[] {
  const input = [...x, Math.log(sigma)];
  const h1 = linearLayer(input,  weights['net.0.weight'], weights['net.0.bias']).map(tanh);
  const h2 = linearLayer(h1,     weights['net.2.weight'], weights['net.2.bias']).map(tanh);
  return      linearLayer(h2,    weights['net.4.weight'], weights['net.4.bias']);
}

/** Load and parse score-weights.json. Returns null on failure. */
export async function loadScoreWeights(url: string): Promise<ScoreNetWeights | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json() as ScoreNetWeights;
  } catch {
    return null;
  }
}
