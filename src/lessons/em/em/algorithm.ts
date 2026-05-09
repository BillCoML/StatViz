import { TRIALS, FLIPS_PER_TRIAL, PRIOR } from './data';

export interface EMState {
  thetaA: number;
  thetaB: number;
  iteration: number;
  responsibilities: { gammaA: number; gammaB: number }[];
  observedLogLikelihood: number;
}

const EPS = 1e-12;
const clamp = (t: number) => Math.max(EPS, Math.min(1 - EPS, t));

function trialLogWeight(theta: number, heads: number, tails: number): number {
  const t = clamp(theta);
  return heads * Math.log(t) + tails * Math.log(1 - t);
}

export function eStep(thetaA: number, thetaB: number) {
  return TRIALS.map(({ heads, tails }) => {
    const lA = Math.log(PRIOR) + trialLogWeight(thetaA, heads, tails);
    const lB = Math.log(1 - PRIOR) + trialLogWeight(thetaB, heads, tails);
    const m = Math.max(lA, lB);
    const a = Math.exp(lA - m);
    const b = Math.exp(lB - m);
    const gammaA = a / (a + b);
    return { gammaA, gammaB: 1 - gammaA };
  });
}

export function mStep(R: { gammaA: number; gammaB: number }[]): { thetaA: number; thetaB: number } {
  let nA = 0, dA = 0, nB = 0, dB = 0;
  R.forEach(({ gammaA, gammaB }, i) => {
    const { heads } = TRIALS[i];
    nA += gammaA * heads;
    dA += gammaA * FLIPS_PER_TRIAL;
    nB += gammaB * heads;
    dB += gammaB * FLIPS_PER_TRIAL;
  });
  return { thetaA: nA / dA, thetaB: nB / dB };
}

export function observedLogLikelihood(thetaA: number, thetaB: number): number {
  return TRIALS.reduce((acc, { heads, tails }) => {
    const lA = Math.log(PRIOR) + trialLogWeight(thetaA, heads, tails);
    const lB = Math.log(1 - PRIOR) + trialLogWeight(thetaB, heads, tails);
    const m = Math.max(lA, lB);
    return acc + m + Math.log(Math.exp(lA - m) + Math.exp(lB - m));
  }, 0);
}

export function initialState(thetaA0: number, thetaB0: number): EMState {
  const R = eStep(thetaA0, thetaB0);
  return {
    thetaA: thetaA0,
    thetaB: thetaB0,
    iteration: 0,
    responsibilities: R,
    observedLogLikelihood: observedLogLikelihood(thetaA0, thetaB0),
  };
}

export function runEMStep(s: EMState): EMState {
  const R = eStep(s.thetaA, s.thetaB);
  const { thetaA, thetaB } = mStep(R);
  return {
    thetaA,
    thetaB,
    iteration: s.iteration + 1,
    responsibilities: eStep(thetaA, thetaB),
    observedLogLikelihood: observedLogLikelihood(thetaA, thetaB),
  };
}

export function runUntilConvergence(
  thetaA0: number,
  thetaB0: number,
  maxIter = 200,
  tol = 1e-8,
): EMState[] {
  const history = [initialState(thetaA0, thetaB0)];
  for (let t = 0; t < maxIter; t++) {
    const last = history[history.length - 1];
    const next = runEMStep(last);
    history.push(next);
    if (Math.abs(next.thetaA - last.thetaA) + Math.abs(next.thetaB - last.thetaB) < tol) break;
  }
  return history;
}

/** Q(θ | θ^(t)) — expected complete-data log-likelihood */
export function qFunction(
  thetaA: number,
  thetaB: number,
  responsibilities: { gammaA: number; gammaB: number }[],
): number {
  return TRIALS.reduce((acc, { heads, tails }, i) => {
    const { gammaA, gammaB } = responsibilities[i];
    return (
      acc +
      gammaA * trialLogWeight(thetaA, heads, tails) +
      gammaB * trialLogWeight(thetaB, heads, tails)
    );
  }, 0);
}

/** H(θ | θ^(t)) = E_{Z~k(·|x,θ^(t))}[log k(Z | x, θ)] */
export function hFunction(
  thetaA: number,
  thetaB: number,
  responsibilities: { gammaA: number; gammaB: number }[],
): number {
  return TRIALS.reduce((acc, { heads, tails }, i) => {
    const { gammaA, gammaB } = responsibilities[i];
    const lA = Math.log(PRIOR) + trialLogWeight(thetaA, heads, tails);
    const lB = Math.log(1 - PRIOR) + trialLogWeight(thetaB, heads, tails);
    const m = Math.max(lA, lB);
    const sumExp = Math.exp(lA - m) + Math.exp(lB - m);
    const logNorm = m + Math.log(sumExp);
    const logKA = lA - logNorm;
    const logKB = lB - logNorm;
    return acc + gammaA * logKA + gammaB * logKB;
  }, 0);
}
