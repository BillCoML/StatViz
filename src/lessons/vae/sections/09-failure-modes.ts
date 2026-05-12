import { renderMath, callout } from '@shared/ui';
import { mountBetaVAESlider } from '../viz/beta-vae-slider';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-9';
  sec.className = 'section';
  sec.innerHTML = `
    <div id="posterior-collapse"></div>
    <div class="section-label">§9</div>
    <h2>Failure Modes</h2>
    <div class="prose">
      <h3>Posterior collapse</h3>
      <p>A pathology where the encoder collapses to the prior:</p>
      $$q_\\phi(z \\mid x) \\;\\approx\\; p(z) \\;=\\; \\mathcal{N}(0, I) \\quad
        \\text{for every } x$$
      <p>Then <strong>$z$ contains no information about $x$</strong>. The decoder ignores
      $z$ and outputs the mean of the data distribution. The KL is at its minimum (zero)
      and the reconstruction has bottomed out at what a non-informative $z$ achieves.</p>

      <p><strong>Why this happens</strong>: optimization can reach a "lazy" minimum
      where the encoder stops encoding and the decoder learns a constant. Especially
      likely when the decoder is very expressive.</p>

      <p><strong>Symptoms</strong>:</p>
      <ul>
        <li>KL loss near zero throughout training.</li>
        <li>Reconstructions are blurry / generic.</li>
        <li>Sampling from the prior produces "average" outputs only.</li>
      </ul>

      <h3>Reconstruction collapse (the opposite pathology)</h3>
      <p>The encoder overfits: each $x$ maps to a near-delta $q_\\phi(z \\mid x)$,
      the decoder reconstructs perfectly, but the latent space is so fragmented that
      prior samples decode to nothing meaningful.</p>

      <p><strong>Why this happens</strong>: when the KL regularizer is too weak
      (e.g., $\\beta$ too small), the model trades regularization for reconstruction quality.</p>

      <h3>β-VAE: tuning the trade-off</h3>
      <p>Scale the KL term by a hyperparameter $\\beta$:</p>
      $$\\mathrm{ELBO}_\\beta(x) \\;=\\;
        \\mathbb{E}_{q_\\phi(z \\mid x)}[\\log p_\\theta(x \\mid z)]
        \\;-\\; \\beta \\, D_{\\mathrm{KL}}\\!\\big(q_\\phi(z \\mid x) \\,\\|\\, p(z)\\big)$$
      <ul>
        <li>$\\beta < 1$: weaker regularization → better reconstruction, worse sampling. Drift toward reconstruction collapse.</li>
        <li>$\\beta = 1$: the standard ELBO. The principled choice.</li>
        <li>$\\beta > 1$: stronger regularization → "disentangled" representations (Higgins et al. 2017), but worse reconstruction. Drift toward posterior collapse.</li>
      </ul>
      <p>Note: for $\\beta \\neq 1$, this is no longer the ELBO of any model — but it is a useful knob.</p>
    </div>

    ${callout('tip', 'Watching the β sweep on this dataset',
      `<p>Set the β slider in §8 to 10: all four cluster means collapse to ~(0,0)
      in latent space — this is the posterior collapse shown live.
      Set β to 0.25: clusters spread out past radius 1 in latent space and
      prior samples decode to regions outside the training data.
      β = 1 shows the balanced case where clusters are distinct but well-regularized.</p>`
    )}

    <div id="viz-beta-vae-slider" class="viz-medium"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountBetaVAESlider(sec.querySelector('#viz-beta-vae-slider') as HTMLElement);
}
