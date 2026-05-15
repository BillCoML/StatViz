import { describe, it, expect } from 'vitest';
import { timeEmbed, epsNetForward, type DDPMWeights } from './eps-net';

describe('timeEmbed', () => {
  it('first half is sin, second half is cos, with t=0 → (0..0, 1..1)', () => {
    const emb = timeEmbed(0, 32);
    expect(emb).toHaveLength(32);
    for (let i = 0; i < 16; i++) expect(emb[i]).toBeCloseTo(0, 12);
    for (let i = 16; i < 32; i++) expect(emb[i]).toBeCloseTo(1, 12);
  });

  it('different t produces different embeddings', () => {
    const e0 = timeEmbed(10, 32);
    const e1 = timeEmbed(50, 32);
    let diff = 0;
    for (let i = 0; i < 32; i++) diff += Math.abs(e0[i] - e1[i]);
    expect(diff).toBeGreaterThan(1);
  });
});

describe('epsNetForward', () => {
  // Minimal weights: hidden=4, time_dim=4 (so half=2, no div-by-zero in freqs).
  const inDim = 2 + 4;
  const eye = (n: number, m: number) =>
    Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) => i === j ? 1 : 0));
  const W: DDPMWeights = {
    'net.0.weight': eye(4, inDim), 'net.0.bias': [0,0,0,0],
    'net.2.weight': eye(4, 4),     'net.2.bias': [0,0,0,0],
    'net.4.weight': eye(4, 4),     'net.4.bias': [0,0,0,0],
    'net.6.weight': eye(2, 4),     'net.6.bias': [0,0],
    _metadata: { T: 10, betas: [], alpha_bars: [], data_centers: [], hidden_dim: 4, time_dim: 4, epochs: 0 },
  };

  it('runs without error and returns 2D output', () => {
    const out = epsNetForward([0.5, -0.3], 5, W);
    expect(out).toHaveLength(2);
    expect(Number.isFinite(out[0])).toBe(true);
    expect(Number.isFinite(out[1])).toBe(true);
  });
});
