import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-4';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'ism');
  sec.innerHTML = `
    <div class="section-label">§4</div>
    <h2>Implicit Score Matching (Hyvärinen)</h2>
    <div class="prose">
      <p>The first route to a tractable score-matching loss is to <strong>rewrite
      the inaccessible term using integration by parts.</strong> This was Hyvärinen's
      original (2005) contribution.</p>

      <p>Expand the squared norm in $\\mathcal{L}_{\\mathrm{SM}}$:</p>

      $$\\mathcal{L}_{\\mathrm{SM}} = \\mathbb{E}_{p_{\\mathrm{data}}}\\!\\left[\\|s_\\theta(x)\\|^2\\right]
      - 2\\, \\mathbb{E}_{p_{\\mathrm{data}}}\\!\\left[s_\\theta(x)^\\top \\nabla_x \\log p_{\\mathrm{data}}(x)\\right]
      + \\underbrace{\\mathbb{E}_{p_{\\mathrm{data}}}\\!\\left[\\|\\nabla_x \\log p_{\\mathrm{data}}(x)\\|^2\\right]}_{\\text{constant in } \\theta}$$

      <p>Drop the third term. The first term is easy. The middle term is the problem.</p>

      <h3>The integration-by-parts trick</h3>
      <p>Use the identity $p_{\\mathrm{data}}(x)\\, \\nabla \\log p_{\\mathrm{data}}(x) =
      \\nabla p_{\\mathrm{data}}(x)$:</p>

      $$\\mathbb{E}_{p_{\\mathrm{data}}}\\!\\left[s_\\theta(x)^\\top \\nabla \\log p_{\\mathrm{data}}(x)\\right]
      = \\int s_\\theta(x)^\\top \\nabla p_{\\mathrm{data}}(x)\\, dx$$

      <p>Integrate by parts in each coordinate $i$:</p>

      $$\\int s_{\\theta,i}(x)\\, \\frac{\\partial p_{\\mathrm{data}}}{\\partial x_i}\\, dx
      = \\underbrace{\\left[s_{\\theta,i}(x)\\, p_{\\mathrm{data}}(x)\\right]_{-\\infty}^{\\infty}}_{= 0}
      - \\int p_{\\mathrm{data}}(x)\\, \\frac{\\partial s_{\\theta,i}}{\\partial x_i}\\, dx$$

      <p>The boundary term vanishes: $p_{\\mathrm{data}}(x) \\to 0$ as $\\|x\\| \\to \\infty$
      (any reasonable data distribution) and $s_\\theta$ doesn't grow too fast (any neural
      network with bounded weights). Summing over $i$:</p>

      $$\\mathbb{E}_{p_{\\mathrm{data}}}\\!\\left[s_\\theta(x)^\\top \\nabla \\log p_{\\mathrm{data}}(x)\\right]
      = -\\mathbb{E}_{p_{\\mathrm{data}}}\\!\\left[\\mathrm{tr}\\!\\left(\\nabla s_\\theta(x)\\right)\\right]$$

      <p>The trace of the Jacobian of $s_\\theta$. <strong>No more $\\nabla \\log p_{\\mathrm{data}}$.</strong>
      Substituting back:</p>

      <div class="formula-box">
        $$\\mathcal{L}_{\\mathrm{ISM}}(\\theta) = \\mathbb{E}_{x \\sim p_{\\mathrm{data}}}\\!\\left[
        \\|s_\\theta(x)\\|^2 + 2\\,\\mathrm{tr}\\!\\left(\\nabla s_\\theta(x)\\right)\\right]$$
      </div>

      <p>Up to the constant we dropped, this <strong>equals</strong>
      $\\mathcal{L}_{\\mathrm{SM}}$, but every term is computable from data samples alone.</p>

      <h3>Sanity check</h3>
      <p>Let $p_{\\mathrm{data}} = \\mathcal{N}(0, I_d)$ and parameterize
      $s_\\theta(x) = ax$ for a single scalar $a$. Then
      $\\|s_\\theta\\|^2 = a^2\\|x\\|^2$ and $\\mathrm{tr}(\\nabla s_\\theta) = ad$. So:</p>

      $$\\mathcal{L}_{\\mathrm{ISM}} = a^2 \\cdot d + 2ad = d(a^2 + 2a)$$

      <p>Minimizing over $a$: derivative is $d(2a + 2) = 0$, so $a = -1$.
      This recovers $s_\\theta(x) = -x$, the <strong>exact</strong> score of
      $\\mathcal{N}(0, I_d)$. The framework works.</p>

      <h3>The catch: trace cost</h3>
      <p>Computing $\\mathrm{tr}(\\nabla s_\\theta(x))$ exactly requires $d$ backward passes
      (one per coordinate). For image data with $d = 10^5$ or more, this is prohibitive.</p>

      <p>Two ways around it:</p>
      <ul>
        <li><strong>Sliced score matching:</strong> replace the trace with a Hutchinson
        trace estimator using random projections. $O(1)$ extra cost per step.</li>
        <li><strong>Denoising score matching:</strong> avoid the trace entirely by changing
        what we're matching against. <strong>§5.</strong></li>
      </ul>

      <p>The DSM route is what generative diffusion models use.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'mvn-density',
      toAnchorLabel: '2 — MVN density',
      body: `The sanity-check example above uses the fact that the score of
             $\\mathcal{N}(0, I)$ is $s(x) = -x$ — a direct corollary of the
             Cookbook's multivariate Gaussian density formula.`,
    })}

    <div id="viz-ism-derivation" class="viz-placeholder">
      <p class="viz-placeholder__label">Visualization: ISM Derivation (step-through)</p>
    </div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
