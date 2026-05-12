import { describe, it, expect } from 'vitest';
import { mlpForward, encoderForward, decoderForward, reparameterize } from './encoder-decoder';
import { TRACE_ENCODER_WEIGHTS, TRACE_DECODER_WEIGHTS } from './trace-weights';

describe('mlpForward', () => {
  it('is deterministic on repeated calls', () => {
    const wts = { W1: [[0.5, -0.3], [0.2, 0.4]], b1: [0.1, -0.1], W2: [[0.4, 0.2], [-0.3, 0.1]], b2: [0.0, 0.5] };
    const r1 = mlpForward([1.0, -1.0], wts);
    const r2 = mlpForward([1.0, -1.0], wts);
    expect(r1).toEqual(r2);
  });

  it('computes tanh hidden layer then linear output', () => {
    const w = {
      W1: [[1.0, 0.0], [0.0, 1.0]],
      b1: [0.0, 0.0],
      W2: [[1.0, 0.0], [0.0, 1.0]],
      b2: [0.0, 0.0],
    };
    const [y0, y1] = mlpForward([0.5, -0.5], w);
    expect(y0).toBeCloseTo(Math.tanh(0.5), 10);
    expect(y1).toBeCloseTo(Math.tanh(-0.5), 10);
  });
});

describe('reparameterize', () => {
  it('z = mu + exp(log_sigma) * eps', () => {
    const mu = [0.34, -0.12];
    const ls = [-0.2, -0.4];
    const eps = [0.5, -0.3];
    const z = reparameterize(mu, ls, eps);
    expect(z[0]).toBeCloseTo(0.34 + Math.exp(-0.2) * 0.5, 10);
    expect(z[1]).toBeCloseTo(-0.12 + Math.exp(-0.4) * (-0.3), 10);
  });
});

describe('trace weights — §7 worked example', () => {
  const x = [1.2, -0.8];
  const eps = [0.5, -0.3];

  it('encoder produces mu ≈ [0.34, -0.12]', () => {
    const { mu } = encoderForward(x, TRACE_ENCODER_WEIGHTS, 2);
    expect(mu[0]).toBeCloseTo(0.34, 4);
    expect(mu[1]).toBeCloseTo(-0.12, 4);
  });

  it('encoder produces log_sigma ≈ [-0.20, -0.40]', () => {
    const { log_sigma } = encoderForward(x, TRACE_ENCODER_WEIGHTS, 2);
    expect(log_sigma[0]).toBeCloseTo(-0.2, 4);
    expect(log_sigma[1]).toBeCloseTo(-0.4, 4);
  });

  it('decoder produces mu_theta ≈ [0.95, -0.71]', () => {
    const { mu, log_sigma } = encoderForward(x, TRACE_ENCODER_WEIGHTS, 2);
    const z = reparameterize(mu, log_sigma, eps);
    const mu_theta = decoderForward(z, TRACE_DECODER_WEIGHTS);
    expect(mu_theta[0]).toBeCloseTo(0.95, 3);
    expect(mu_theta[1]).toBeCloseTo(-0.71, 3);
  });

  it('full forward pass is deterministic', () => {
    const run = () => {
      const { mu, log_sigma } = encoderForward(x, TRACE_ENCODER_WEIGHTS, 2);
      const z = reparameterize(mu, log_sigma, eps);
      return decoderForward(z, TRACE_DECODER_WEIGHTS);
    };
    const r1 = run();
    const r2 = run();
    expect(r1[0]).toBeCloseTo(r2[0], 10);
    expect(r1[1]).toBeCloseTo(r2[1], 10);
  });
});
