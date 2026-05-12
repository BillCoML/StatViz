import { renderMath, crosslinkForward } from '@shared/ui';
import { mountReparamFlow } from '../viz/reparam-flow';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-4';
  sec.className = 'section';
  sec.dataset.anchor = 'reparam-matrix';
  sec.innerHTML = `
    <div id="reparam-matrix"></div>
    <div class="section-label">§4</div>
    <h2>The Reparameterization Trick (Matrix Form)</h2>

    <div class="prose">
      $$\\boxed{\\;\\; Z \\sim \\mathcal{N}(\\mu, \\Sigma)
        \\;\\iff\\; Z = \\mu + L \\, \\varepsilon, \\;\\;
        \\varepsilon \\sim \\mathcal{N}(0, I), \\;\\;
        L L^\\top = \\Sigma \\;\\;}$$

      <p>To sample $Z \\sim \\mathcal{N}(\\mu, \\Sigma)$ in a way that's
      <strong>differentiable with respect to $\\mu$ and $\\Sigma$</strong>, factor $\\Sigma$
      as $L L^\\top$ (where $L$ is, e.g., the Cholesky factor), draw a standard normal
      $\\varepsilon \\sim \\mathcal{N}(0, I)$, and form $Z = \\mu + L\\varepsilon$.</p>

      <p>Why this works: $\\varepsilon$ has zero mean and identity covariance, so by the
      affine-transformation rule from §2,</p>
      $$\\mathbb{E}[Z] = \\mu, \\qquad \\mathrm{Cov}(Z) = L \\cdot I \\cdot L^\\top = \\Sigma$$
      <p>$Z$ is Gaussian (affine of a Gaussian), with the right mean and covariance.</p>

      <h3>Why the trick matters</h3>
      <p>The point isn't to sample (which we could do directly). The point is that the
      <strong>randomness has been separated from the parameters</strong>. $\\varepsilon$
      has no parameters; $\\mu$ and $L$ are deterministic. So for any smooth $f$:</p>
      $$\\nabla_{\\mu, L} \\, \\mathbb{E}_{Z \\sim \\mathcal{N}(\\mu, \\Sigma)}[f(Z)]
        \\;=\\; \\mathbb{E}_{\\varepsilon}\\!\\left[\\nabla_{\\mu, L} f(\\mu + L\\varepsilon)\\right]$$
      <p>Gradients pass through the sampling step. <strong>This is what makes end-to-end
      gradient training of VAEs possible.</strong></p>

      <h3>Diagonal special case (the one VAEs use)</h3>
      <p>If $\\Sigma = \\mathrm{diag}(\\sigma_1^2, \\ldots, \\sigma_d^2)$, then
      $L = \\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_d)$ and the rule collapses to
      element-wise:</p>
      $$Z_i \\;=\\; \\mu_i + \\sigma_i \\varepsilon_i, \\;\\; \\varepsilon_i \\sim \\mathcal{N}(0, 1)$$
      <p>In vector notation: $Z = \\mu + \\sigma \\odot \\varepsilon$ where $\\odot$ is
      element-wise multiplication. One line of code in any VAE implementation.</p>

      <div class="worked-example">
        <div class="worked-example-title">Worked numerical example</div>
        <p>Take $\\mu = (1, -1)$ and $\\Sigma = \\bigl(\\begin{smallmatrix} 4 & 1 \\\\ 1 & 1 \\end{smallmatrix}\\bigr)$.
        The Cholesky factor is</p>
        $$L \\;=\\; \\begin{pmatrix} 2 & 0 \\\\ 0.5 & \\sqrt{0.75} \\end{pmatrix}
          \\;\\approx\\; \\begin{pmatrix} 2 & 0 \\\\ 0.5 & 0.866 \\end{pmatrix}$$
        <p>Drawing $\\varepsilon = (0.3, -1.2)$ gives</p>
        $$Z = (1, -1) + (2 \\cdot 0.3, \\; 0.5 \\cdot 0.3 + 0.866 \\cdot (-1.2)) = (1.6, -1.890)$$
        <p>The transformation is deterministic given $\\varepsilon$; gradients with respect
        to $\\mu$ and $L$ pass straight through.</p>
      </div>
    </div>

    <div id="viz-reparam-flow" class="viz-wide"></div>

    ${crosslinkForward({
      toLesson: 'vae',
      toAnchor: 'reparam-in-vae',
      body: `<p>The encoder outputs $(\\mu_\\phi(x), \\sigma_\\phi^2(x))$; the sample
        $z = \\mu + \\sigma \\odot \\varepsilon$ is differentiable end-to-end through
        the encoder weights.</p>`,
    })}

    ${crosslinkForward({
      toLesson: 'ddpm',
      body: `<p>Noise is added via $x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1 - \\bar\\alpha_t} \\varepsilon$,
        which is the reparameterization trick applied to the diffusion forward process.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountReparamFlow(sec.querySelector('#viz-reparam-flow') as HTMLElement);
}
