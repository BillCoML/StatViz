import { renderMath, crosslinkBack } from '@shared/ui';
import { mountLossDecomposition } from '../viz/loss-decomposition';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-4';
  sec.className = 'section';
  sec.innerHTML = `
    <div id="vae-objective"></div>
    <div class="section-label">§4</div>
    <h2>The VAE Objective</h2>
    <div class="prose">
      <p>Write out the per-example ELBO using
      <a href="/StatViz/lessons/elbo-vi/#elbo-two-forms">Form 2</a>
      from ELBO &amp; VI §4 (reconstruction minus KL):</p>

      $$\\boxed{\\;\\;\\mathrm{ELBO}(x; \\theta, \\phi) \\;=\\;
        \\underbrace{\\mathbb{E}_{q_\\phi(z \\mid x)}\\!\\left[\\log p_\\theta(x \\mid z)\\right]}_{\\text{reconstruction}}
        \\;-\\;
        \\underbrace{D_{\\mathrm{KL}}\\!\\big(q_\\phi(z \\mid x) \\,\\|\\, p(z)\\big)}_{\\text{regularizer}}
      \\;\\;}$$

      <h3>Reconstruction term</h3>
      $$\\mathbb{E}_{q_\\phi(z \\mid x)}\\!\\big[\\log p_\\theta(x \\mid z)\\big]$$
      <p>Sample $z$ from the encoder's distribution; ask how well the decoder's
      distribution explains $x$ when conditioned on that $z$.
      For a Gaussian decoder $p_\\theta(x \\mid z) = \\mathcal{N}(\\mu_\\theta(z), \\sigma_x^2 I)$,
      this reduces (up to a constant) to:</p>
      $$-\\frac{1}{2\\sigma_x^2}\\,\\mathbb{E}_{q_\\phi}\\!\\big[\\|x - \\mu_\\theta(z)\\|^2\\big]$$
      <p>Maximizing the reconstruction term is <strong>minimizing expected squared error</strong>
      between $x$ and the decoder's prediction.</p>

      <h3>Regularizer term</h3>
      $$D_{\\mathrm{KL}}\\!\\big(q_\\phi(z \\mid x) \\,\\|\\, p(z)\\big)$$
      <p>The reverse KL between the encoder's posterior approximation and the prior
      $p(z) = \\mathcal{N}(0, I)$. Minimizing this term
      <strong>pulls every $q_\\phi(z \\mid x)$ toward the standard normal</strong>:</p>
      <ul>
        <li>At sampling time we generate $z \\sim p(z)$ and decode. For decoded samples
        to look like real data, $z$-values from the prior need to match those the encoder
        produced from real data. The regularizer ensures this overlap.</li>
        <li>Without the regularizer, the encoder can cheat by mapping each $x$ to a
        delta function — maximizing reconstruction while destroying structure.</li>
      </ul>

      <h3>The trade-off</h3>
      <p>The two terms pull in opposite directions:</p>
      <ul>
        <li><strong>Reconstruction</strong> wants $q_\\phi(z \\mid x)$ to be sharp
        around a $z$ the decoder maps reliably back to $x$.</li>
        <li><strong>Regularizer</strong> wants $q_\\phi(z \\mid x)$ to be near
        $\\mathcal{N}(0, I)$ — broad and generic.</li>
      </ul>
      <p>The trained model finds a compromise. Use the visualization below to
      build intuition before seeing the real thing in §8.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'elbo-vi',
      toAnchor: 'elbo-two-forms',
      toAnchorLabel: '4',
      body: `<p>The boxed objective above is exactly Form 2 of the ELBO from ELBO &amp; VI,
        specialized to the VAE setting where the prior is $\\mathcal{N}(0,I)$ and the
        variational family is the encoder network. No new derivation needed.</p>`,
    })}

    <div id="viz-loss-decomposition" class="viz-medium"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountLossDecomposition(sec.querySelector('#viz-loss-decomposition') as HTMLElement);
}
