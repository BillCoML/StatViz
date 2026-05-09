/** Numerically compute the Jensen gap E[phi(X)] - phi(E[X]) for a discrete distribution. */
export function jensenGap(values: number[], probs: number[],
                          phi: (x: number) => number): {
  expectationOfPhi: number;
  phiOfExpectation: number;
  gap: number;
} {
  let ex = 0, ephi = 0;
  for (let i = 0; i < values.length; i++) {
    ex   += values[i] * probs[i];
    ephi += phi(values[i]) * probs[i];
  }
  return {
    expectationOfPhi: ephi,
    phiOfExpectation: phi(ex),
    gap: ephi - phi(ex),
  };
}
