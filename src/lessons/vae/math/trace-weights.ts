import type { MLPWeights } from './encoder-decoder';

/**
 * Hand-crafted encoder weights for the §7 numerical trace.
 * Designed so that encoder([1.2, -0.8]) produces
 *   mu = [0.34, -0.12],  log_sigma = [-0.20, -0.40].
 *
 * Construction: b1 is chosen so W1@x+b1 = [atanh(0.5), atanh(-0.5), …]
 * for x = [1.2, -0.8], giving h = [0.5, -0.5, 0.5, -0.5].
 * Then W2 = I and b2 = target - h.
 */
export const TRACE_ENCODER_WEIGHTS: MLPWeights = {
  W1: [
    [ 0.3,  0.2],
    [-0.2,  0.1],
    [ 0.1,  0.3],
    [-0.3,  0.1],
  ],
  //  atanh(0.5) = 0.5493061443340549
  //  Row 0: need W1[0]@x + b1[0] = atanh( 0.5)  =>  0.20 + b1[0] = 0.5493…  =>  b1[0] = 0.3493…
  //  Row 1: need W1[1]@x + b1[1] = atanh(-0.5)  => -0.32 + b1[1] = -0.5493… => b1[1] = -0.2293…
  //  Row 2: need W1[2]@x + b1[2] = atanh( 0.5)  => -0.12 + b1[2] = 0.5493…  => b1[2] = 0.6693…
  //  Row 3: need W1[3]@x + b1[3] = atanh(-0.5)  => -0.44 + b1[3] = -0.5493… => b1[3] = -0.1093…
  b1: [0.3493061443, -0.2293061443, 0.6693061443, -0.1093061443],
  W2: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
  // h = [0.5, -0.5, 0.5, -0.5]  →  b2 = target - h
  // target = [0.34, -0.12, -0.20, -0.40]
  b2: [-0.16, 0.38, -0.70, 0.10],
};

/**
 * Hand-crafted decoder weights for the §7 trace.
 * Designed so that decoder([0.74937, -0.32110]) produces [0.95, -0.71],
 * where z = reparameterize([0.34,-0.12], [-0.2,-0.4], [0.5,-0.3]).
 *
 * b1 chosen so W1@z+b1 = [atanh(0.5), atanh(-0.5), atanh(0.5), atanh(-0.5)].
 * W2 picks the first two hidden units; b2 = target - [0.5, -0.5].
 */
export const TRACE_DECODER_WEIGHTS: MLPWeights = {
  W1: [
    [ 0.2,  0.3],
    [-0.3,  0.2],
    [ 0.1, -0.2],
    [ 0.3,  0.1],
  ],
  // z ≈ [0.749365, -0.321096]  (exact: 0.34+exp(-0.2)*0.5, -0.12+exp(-0.4)*(-0.3))
  // b1[0] = atanh(0.5) - (0.2*z[0] + 0.3*z[1]) ≈ 0.4957618732
  // b1[1] = atanh(-0.5) - (-0.3*z[0] + 0.2*z[1]) ≈ -0.2602773286
  // b1[2] = atanh(0.5) - (0.1*z[0] - 0.2*z[1]) ≈ 0.4101503039
  // b1[3] = atanh(-0.5) - (0.3*z[0] + 0.1*z[1]) ≈ -0.7420061559
  b1: [0.4957618732, -0.2602773286, 0.4101503039, -0.7420061559],
  W2: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
  ],
  // h = [0.5, -0.5, 0.5, -0.5]  →  b2 = [0.95-0.5, -0.71-(-0.5)] = [0.45, -0.21]
  b2: [0.45, -0.21],
};
