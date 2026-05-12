export interface MLPWeights {
  W1: number[][];  // hidden_dim × input_dim
  b1: number[];    // hidden_dim
  W2: number[][];  // output_dim × hidden_dim
  b2: number[];    // output_dim
}

export function mlpForward(input: number[], weights: MLPWeights): number[] {
  const h = weights.W1.map((row, i) =>
    Math.tanh(row.reduce((s, w, j) => s + w * input[j], 0) + weights.b1[i])
  );
  return weights.W2.map((row, i) =>
    row.reduce((s, w, j) => s + w * h[j], 0) + weights.b2[i]
  );
}

export type EncoderWeights = MLPWeights;

/** Encoder forward pass — output is split into (mu, log_sigma) each of size latent_dim. */
export function encoderForward(
  x: number[],
  w: EncoderWeights,
  latent_dim: number,
): { mu: number[]; log_sigma: number[] } {
  const out = mlpForward(x, w);
  return { mu: out.slice(0, latent_dim), log_sigma: out.slice(latent_dim) };
}

/** Decoder forward pass — returns mu_theta(z). */
export function decoderForward(z: number[], w: MLPWeights): number[] {
  return mlpForward(z, w);
}

/** z = mu + exp(log_sigma) ⊙ eps */
export function reparameterize(mu: number[], log_sigma: number[], eps: number[]): number[] {
  return mu.map((m, i) => m + Math.exp(log_sigma[i]) * eps[i]);
}
