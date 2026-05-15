import { renderMath, crosslinkForward } from '@shared/ui';
import { mountMVNExplorer } from '../viz/mvn-explorer';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-2';
  sec.className = 'section';
  sec.dataset.anchor = 'mvn-density';
  sec.innerHTML = `
    <div id="mvn-density"></div>
    <div class="section-label">§2</div>
    <h2>The Multivariate Gaussian: Foundations</h2>
    <div class="prose">
      <p>A $d$-dimensional random vector $X \\in \\mathbb{R}^d$ is <strong>multivariate
      Gaussian</strong> with mean $\\mu \\in \\mathbb{R}^d$ and covariance
      $\\Sigma \\in \\mathbb{R}^{d \\times d}$ (symmetric positive definite) if its density is</p>
      $$\\boxed{\\;\\; \\mathcal{N}(x; \\mu, \\Sigma) \\;=\\; \\frac{1}{(2\\pi)^{d/2} \\, |\\Sigma|^{1/2}} \\, \\exp\\!\\left(-\\tfrac{1}{2} (x - \\mu)^\\top \\Sigma^{-1} (x - \\mu)\\right) \\;\\;}$$
      <p>where $|\\Sigma|$ is the determinant. The <strong>log density</strong> drops the
      exponential and the constants:</p>
      $$\\log \\mathcal{N}(x; \\mu, \\Sigma) \\;=\\; -\\tfrac{d}{2} \\log(2\\pi)
        \\;-\\; \\tfrac{1}{2} \\log |\\Sigma|
        \\;-\\; \\tfrac{1}{2} (x - \\mu)^\\top \\Sigma^{-1} (x - \\mu)$$
      <p>The quadratic form $(x - \\mu)^\\top \\Sigma^{-1} (x - \\mu)$ is the
      <strong>squared Mahalanobis distance</strong> between $x$ and $\\mu$ under the
      metric $\\Sigma^{-1}$.</p>

      <h3>Three special cases worth memorizing</h3>
      <ol>
        <li><strong>Isotropic</strong> ($\\Sigma = \\sigma^2 I$): contours are spheres,
        density depends only on $\\|x - \\mu\\|^2$:
        $$\\mathcal{N}(x; \\mu, \\sigma^2 I) \\;\\propto\\; \\exp\\!\\left(-\\frac{\\|x - \\mu\\|^2}{2\\sigma^2}\\right)$$
        </li>
        <li><strong>Diagonal</strong> ($\\Sigma = \\mathrm{diag}(\\sigma_1^2, \\ldots, \\sigma_d^2)$):
        coordinates are independent, density factorizes:
        $$\\mathcal{N}(x; \\mu, \\Sigma) \\;=\\; \\prod_{i=1}^{d} \\mathcal{N}(x_i; \\mu_i, \\sigma_i^2)$$
        </li>
        <li><strong>Standard</strong> ($\\mu = 0, \\Sigma = I$): the reference distribution,
        used as the prior in nearly every VAE and as the noise distribution in DDPM.
        </li>
      </ol>

      <h3>Affine transformations preserve Gaussianity</h3>
      <p>If $X \\sim \\mathcal{N}(\\mu, \\Sigma)$ and $Y = AX + b$ for
      $A \\in \\mathbb{R}^{m \\times d}$ and $b \\in \\mathbb{R}^m$, then</p>
      $$Y \\;\\sim\\; \\mathcal{N}(A\\mu + b, \\; A \\Sigma A^\\top)$$
      <p>The means transform linearly; the covariance transforms as a bilinear form.
      This is the structural fact that makes the reparameterization trick (§4) work.</p>
    </div>

    <div id="viz-mvn-explorer" class="viz-wide"></div>

    ${crosslinkForward({
      toLesson: 'score-matching',
      toAnchor: 'score-definition',
      toAnchorLabel: '2 — The Score Function',
      body: `<p>The score function $s(x) = \\nabla_x \\log p(x)$ of a multivariate
        Gaussian is $-\\Sigma^{-1}(x - \\mu)$ — a direct consequence of the density
        formula above. Score Matching §2 derives this and builds the full score-based
        generative modeling framework on top of it.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountMVNExplorer(sec.querySelector('#viz-mvn-explorer') as HTMLElement);
}
