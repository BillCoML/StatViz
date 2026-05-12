import { renderMath } from '@shared/ui';
import { mountTrainingTrace } from '../viz/training-trace';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-7';
  sec.className = 'section';
  sec.innerHTML = `
    <div id="training-step"></div>
    <div class="section-label">§7</div>
    <h2>One Training Step, Traced</h2>
    <div class="prose">
      <p>Let us trace a single training step end-to-end with concrete numbers.
      The architecture: data dim 2, latent dim 2, hidden dim 4 in each network.
      Data point: $x = (1.2, -0.8)$.  Noise sample: $\\varepsilon = (0.5, -0.3)$.
      Decoder noise: $\\sigma_x = 0.1$.</p>

      <h3>Step 1 — Encoder forward pass</h3>
      $$h_e \\;=\\; \\tanh\\!\\left(W_e x + b_e\\right) \\in \\mathbb{R}^4$$
      $$\\big(\\mu_\\phi(x),\\; \\log\\sigma_\\phi(x)\\big) \\;=\\; W_o h_e + b_o, \\quad
        \\mu, \\log\\sigma \\in \\mathbb{R}^2$$
      <p>For the pre-initialized weights (see <code>trace-weights.ts</code>):</p>
      $$\\mu_\\phi(x) \\;\\approx\\; (0.34,\\; -0.12), \\qquad
        \\log\\sigma_\\phi(x) \\;\\approx\\; (-0.20,\\; -0.40)$$

      <h3>Step 2 — Reparameterize</h3>
      $$z \\;=\\; \\mu + e^{\\log\\sigma} \\odot \\varepsilon
        \\;=\\; (0.34,\\; -0.12) + (0.819,\\; 0.670) \\odot (0.5,\\; -0.3)
        \\;=\\; (0.749,\\; -0.321)$$

      <h3>Step 3 — Decoder forward pass</h3>
      $$h_d \\;=\\; \\tanh\\!\\left(W_d z + b_d\\right) \\in \\mathbb{R}^4, \\qquad
        \\mu_\\theta(z) \\;=\\; W_d' h_d + b_d'$$
      <p>For the pre-initialized weights:</p>
      $$\\mu_\\theta(z) \\;\\approx\\; (0.95,\\; -0.71)$$

      <h3>Step 4 — Reconstruction loss</h3>
      $$L_{\\text{recon}} \\;=\\; \\frac{1}{2\\sigma_x^2} \\|x - \\mu_\\theta(z)\\|^2
        \\;=\\; \\frac{1}{0.02}\\big[(1.2-0.95)^2 + (-0.8+0.71)^2\\big]
        \\;=\\; 50 \\cdot 0.0706 \\;=\\; 3.53$$

      <h3>Step 5 — KL loss</h3>
      $$L_{\\mathrm{KL}} \\;=\\; \\tfrac{1}{2}\\sum_i\\!\\left[
        e^{2\\log\\sigma_i} + \\mu_i^2 - 1 - 2\\log\\sigma_i\\right]$$
      $$=\\; \\tfrac{1}{2}\\big[(0.670 + 0.116 - 1 + 0.4) + (0.449 + 0.014 - 1 + 0.8)\\big]
        \\;\\approx\\; 0.224$$

      <h3>Step 6 — Total loss &amp; backprop</h3>
      $$L_{\\text{total}} \\;=\\; L_{\\text{recon}} + L_{\\mathrm{KL}} \\;=\\; 3.53 + 0.224 \\;=\\; 3.75$$
      <p>Differentiate $L$ with respect to all parameters via the chain rule and update.
      The reparameterization trick is what makes $\\partial L / \\partial \\phi$ computable
      — without it, the gradient would die at Step 2 (the sampling step).</p>
    </div>

    <div id="viz-training-trace" class="viz-wide"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountTrainingTrace(sec.querySelector('#viz-training-trace') as HTMLElement);
}
