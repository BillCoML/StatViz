/**
 * Shared helpers for DDPM visualizations.
 */
import { browserSchedule } from '../math/schedule';
import type { Schedule } from '../math/schedule';
import { epsNetForward, loadDDPMWeights } from '../math/eps-net';
import type { DDPMWeights } from '../math/eps-net';

export const WEIGHTS_URL = (window as any).__BASE__
  ? `${(window as any).__BASE__}assets/ddpm-weights.json`
  : new URL('../assets/ddpm-weights.json', import.meta.url).href;

let _weightsCache: DDPMWeights | null = null;
let _weightsPromise: Promise<DDPMWeights | null> | null = null;

export function getWeights(): Promise<DDPMWeights | null> {
  if (_weightsCache) return Promise.resolve(_weightsCache);
  if (!_weightsPromise) {
    _weightsPromise = loadDDPMWeights(WEIGHTS_URL).then(w => {
      _weightsCache = w;
      return w;
    });
  }
  return _weightsPromise;
}

export const DATA_CENTERS: [number, number][] = [[2, 2], [2, -2], [-2, 2], [-2, -2]];
export const DATA_STD = 0.2;
export const DOMAIN: [number, number] = [-4, 4];

/** Seeded LCG RNG. */
export function mkRng(seed: number): () => number {
  let s = seed | 0;
  return () => { s = (Math.imul(1664525, s) + 1013904223) | 0; return (s >>> 0) / 0x100000000; };
}

export function boxMuller(rng: () => number): number {
  let u = rng(); while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export function gauss2(rng: () => number): [number, number] {
  return [boxMuller(rng), boxMuller(rng)];
}

export function generateDataset(seed = 42, nPer = 50): [number, number][] {
  const rng = mkRng(seed);
  const pts: [number, number][] = [];
  for (const c of DATA_CENTERS) {
    for (let i = 0; i < nPer; i++) {
      pts.push([c[0] + DATA_STD * boxMuller(rng), c[1] + DATA_STD * boxMuller(rng)]);
    }
  }
  return pts;
}

/** Schedule used by the trained browser model (T=100). */
export const browserSched: Schedule = browserSchedule();

/** ε_θ for a batch of N points all at the same t. */
export function epsBatch(xs: number[][], t: number, w: DDPMWeights): number[][] {
  return xs.map(x => epsNetForward(x, t, w));
}

/** Linear scale from data domain to pixel domain. */
export interface Scale { (x: number): number; invert(p: number): number; }

export function makeLinearScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain; const [r0, r1] = range;
  const m = (r1 - r0) / (d1 - d0);
  const fn = ((x: number) => r0 + m * (x - d0)) as Scale;
  fn.invert = (p: number) => d0 + (p - r0) / m;
  return fn;
}
