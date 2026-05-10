import { TRIALS, FLIPS_PER_TRIAL } from '@lessons/em/em/data';

export const M = FLIPS_PER_TRIAL;

export function elboTwoCoins(gammas: number[], thetaA: number, thetaB: number): number {
  let s = -TRIALS.length * Math.log(2);
  for (let i = 0; i < TRIALS.length; i++) {
    const { heads: h, tails: t } = TRIALS[i];
    const g = gammas[i];
    s += g * (h * Math.log(thetaA) + t * Math.log(1 - thetaA))
       + (1 - g) * (h * Math.log(thetaB) + t * Math.log(1 - thetaB));
    if (g > 0 && g < 1) {
      s += -g * Math.log(g) - (1 - g) * Math.log(1 - g);
    }
  }
  return s;
}
