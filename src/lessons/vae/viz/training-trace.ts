import { encoderForward, decoderForward, reparameterize } from '../math/encoder-decoder';
import { vaeELBO } from '../math/elbo';
import { TRACE_ENCODER_WEIGHTS, TRACE_DECODER_WEIGHTS } from '../math/trace-weights';

const LATENT_DIM = 2;

interface TraceState {
  x: number[];
  eps: number[];
  mu: number[];
  log_sigma: number[];
  z: number[];
  x_hat: number[];
  recon: number;
  kl: number;
  elbo: number;
  activeStep: number;
}

function computeTrace(x: number[], eps: number[]): TraceState {
  const { mu, log_sigma } = encoderForward(x, TRACE_ENCODER_WEIGHTS, LATENT_DIM);
  const z = reparameterize(mu, log_sigma, eps);
  const x_hat = decoderForward(z, TRACE_DECODER_WEIGHTS);
  const { recon, kl, elbo } = vaeELBO(x, mu, log_sigma, x_hat, 0.1);
  return { x, eps, mu, log_sigma, z, x_hat, recon, kl, elbo, activeStep: -1 };
}

const fmt = (v: number) => v.toFixed(4);
const vec = (v: number[]) => `(${v.map(fmt).join(', ')})`;
const expV = (ls: number[]) => ls.map(l => Math.exp(l));

function renderSteps(container: HTMLElement, state: TraceState): void {
  const sigma = expV(state.log_sigma);

  const steps = [
    {
      label: '1',
      name: 'Encoder forward pass',
      content: `μ_φ(x) = <span class="vae-trace__value">${vec(state.mu)}</span>, log σ_φ(x) = <span class="vae-trace__value">${vec(state.log_sigma)}</span>`,
    },
    {
      label: '2',
      name: 'Reparameterize',
      content: `z = μ + (${sigma.map(fmt).join(', ')}) ⊙ ε = <span class="vae-trace__value">${vec(state.z)}</span>`,
    },
    {
      label: '3',
      name: 'Decoder forward pass',
      content: `μ_θ(z) = <span class="vae-trace__value">${vec(state.x_hat)}</span>`,
    },
    {
      label: '4',
      name: 'Reconstruction loss',
      content: `L_recon = ½σ_x⁻² ‖x − x̂‖² = <span class="vae-trace__value">${fmt(Math.abs(state.recon))}</span>`,
    },
    {
      label: '5',
      name: 'KL loss',
      content: `L_KL = ½ Σ[σᵢ² + μᵢ² − 1 − log σᵢ²] = <span class="vae-trace__value">${fmt(state.kl)}</span>`,
    },
    {
      label: '6',
      name: 'Total loss (negative ELBO)',
      content: `L = L_recon + L_KL = <span class="vae-trace__value">${fmt(Math.abs(state.recon) + state.kl)}</span>`,
    },
  ];

  const stepsEl = container.querySelector('#trace-steps') as HTMLElement;
  stepsEl.innerHTML = steps.map((s, i) => `
    <div class="vae-trace__step${state.activeStep === i || state.activeStep === 99 ? ' active' : ''}" data-step="${i}">
      <div class="vae-trace__step-num">${s.label}</div>
      <div class="vae-trace__step-content${state.activeStep === i ? ' highlighted' : ''}">
        <strong style="font-size:0.85em;color:var(--ink-soft);">${s.name}</strong><br>
        ${s.content}
      </div>
    </div>
  `).join('');
}

export function mountTrainingTrace(container: HTMLElement): void {
  const DEFAULT_X   = [1.2, -0.8];
  const DEFAULT_EPS = [0.5, -0.3];
  let state = computeTrace(DEFAULT_X, DEFAULT_EPS);
  state.activeStep = -1;

  container.innerHTML = `
    <div class="vae-trace">
      <div style="font-family:var(--font-display);font-weight:600;margin-bottom:0.5rem;">
        Forward pass trace — x = (1.2, −0.8), ε = (0.5, −0.3)
      </div>
      <div id="trace-steps" class="vae-trace__steps"></div>
      <div class="vae-trace__controls">
        <button class="viz-btn-sm" id="btn-trace-prev">← Prev</button>
        <button class="viz-btn-sm" id="btn-trace-next">Next →</button>
        <button class="viz-btn-sm" id="btn-trace-all">Show all</button>
        <button class="viz-btn-sm" id="btn-trace-reroll">Re-roll ε</button>
        <button class="viz-btn-sm" id="btn-trace-reset">Reset</button>
      </div>
      <div id="trace-info" style="font-family:var(--font-mono);font-size:0.78em;color:var(--ink-soft);margin-top:0.5rem;">
        Step 1 of 6
      </div>
    </div>
  `;

  const info = container.querySelector('#trace-info') as HTMLElement;

  function setStep(s: number) {
    state.activeStep = s;
    renderSteps(container, state);
    if (s < 0) info.textContent = 'Click "Next" to begin.';
    else if (s === 99) info.textContent = 'All steps complete.';
    else info.textContent = `Step ${s + 1} of 6`;
  }

  renderSteps(container, state);
  setStep(-1);

  container.querySelector('#btn-trace-next')!.addEventListener('click', () => {
    setStep(state.activeStep < 5 ? state.activeStep + 1 : 99);
  });
  container.querySelector('#btn-trace-prev')!.addEventListener('click', () => {
    setStep(state.activeStep > 0 ? state.activeStep - 1 : -1);
  });
  container.querySelector('#btn-trace-all')!.addEventListener('click', () => setStep(99));

  container.querySelector('#btn-trace-reroll')!.addEventListener('click', () => {
    const eps = [Math.random() * 2 - 1, Math.random() * 2 - 1];
    state = computeTrace(DEFAULT_X, eps);
    state.activeStep = 99;
    const titleEl = container.querySelector('.vae-trace > div') as HTMLElement;
    titleEl.textContent = `Forward pass trace — x = (1.2, −0.8), ε = (${eps.map(v => v.toFixed(2)).join(', ')})`;
    renderSteps(container, state);
    info.textContent = 'ε re-rolled. Results updated.';
  });

  container.querySelector('#btn-trace-reset')!.addEventListener('click', () => {
    state = computeTrace(DEFAULT_X, DEFAULT_EPS);
    const titleEl = container.querySelector('.vae-trace > div') as HTMLElement;
    titleEl.textContent = 'Forward pass trace — x = (1.2, −0.8), ε = (0.5, −0.3)';
    setStep(-1);
  });
}
