import { renderMath, proofToggle, crosslinkForward } from '@shared/ui';
import { mountKLMVNExplorer } from '../viz/kl-mvn-explorer';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-3';
  sec.className = 'section';
  sec.dataset.anchor = 'kl-mvn';
  sec.innerHTML = `
    <div id="kl-mvn"></div>
    <div class="section-label">§3</div>
    <h2>KL Divergence Between Multivariate Gaussians</h2>

    <div class="prose">
      $$\\boxed{\\;\\; D_{\\mathrm{KL}}\\!\\big(\\mathcal{N}(\\mu_1, \\Sigma_1) \\,\\big\\|\\, \\mathcal{N}(\\mu_2, \\Sigma_2)\\big)
        \\;=\\; \\tfrac{1}{2}\\!\\left[\\log \\frac{|\\Sigma_2|}{|\\Sigma_1|}
        \\;-\\; d
        \\;+\\; \\mathrm{tr}(\\Sigma_2^{-1} \\Sigma_1)
        \\;+\\; (\\mu_2 - \\mu_1)^\\top \\Sigma_2^{-1} (\\mu_2 - \\mu_1)\\right] \\;\\;}$$

      <p>The univariate version (from <a href="/StatViz/lessons/kl-jensen/#kl-gaussians">KL &amp; Jensen §4</a>)
      had three terms. The multivariate version has the same four terms generalized:</p>
      <ul>
        <li>$\\log |\\Sigma_2| / |\\Sigma_1|$ — log determinant ratio (analogue of
        $\\log \\sigma_2/\\sigma_1$).</li>
        <li>$-d$ — the dimension (analogue of $-\\tfrac{1}{2} \\times 2$).</li>
        <li>$\\mathrm{tr}(\\Sigma_2^{-1} \\Sigma_1)$ — a "size ratio" term.</li>
        <li>$(\\mu_2 - \\mu_1)^\\top \\Sigma_2^{-1} (\\mu_2 - \\mu_1)$ — squared Mahalanobis
        distance between means under $\\Sigma_2^{-1}$.</li>
      </ul>
      <p>All terms are scalars. The overall expression is non-negative (by Gibbs' inequality),
      and zero iff $(\\mu_1, \\Sigma_1) = (\\mu_2, \\Sigma_2)$.</p>
    </div>

    ${proofToggle('Derivation', `
      <p>Start from the definition:</p>
      $$D_{\\mathrm{KL}} \\;=\\; \\mathbb{E}_{X \\sim \\mathcal{N}(\\mu_1, \\Sigma_1)}\\!\\left[\\log \\frac{\\mathcal{N}(X; \\mu_1, \\Sigma_1)}{\\mathcal{N}(X; \\mu_2, \\Sigma_2)}\\right]$$
      <p>Use the log density formula from §2:</p>
      $$\\log \\frac{\\mathcal{N}(x; \\mu_1, \\Sigma_1)}{\\mathcal{N}(x; \\mu_2, \\Sigma_2)}
        \\;=\\; \\tfrac{1}{2}\\log \\frac{|\\Sigma_2|}{|\\Sigma_1|}
        \\;-\\; \\tfrac{1}{2}(x - \\mu_1)^\\top \\Sigma_1^{-1}(x - \\mu_1)
        \\;+\\; \\tfrac{1}{2}(x - \\mu_2)^\\top \\Sigma_2^{-1}(x - \\mu_2)$$
      <p>Take expectation under $X \\sim \\mathcal{N}(\\mu_1, \\Sigma_1)$.
      The log-determinant term is constant. For the two quadratic forms, use:</p>
      $$\\mathbb{E}[(X - a)^\\top M (X - a)] \\;=\\; \\mathrm{tr}(M \\Sigma_1) \\;+\\; (\\mu_1 - a)^\\top M (\\mu_1 - a)$$
      <p>Applying to the first quadratic with $a = \\mu_1, M = \\Sigma_1^{-1}$:</p>
      $$\\mathbb{E}[(X - \\mu_1)^\\top \\Sigma_1^{-1}(X - \\mu_1)] \\;=\\; \\mathrm{tr}(\\Sigma_1^{-1} \\Sigma_1) + 0 \\;=\\; d$$
      <p>Applying to the second with $a = \\mu_2, M = \\Sigma_2^{-1}$:</p>
      $$\\mathbb{E}[(X - \\mu_2)^\\top \\Sigma_2^{-1}(X - \\mu_2)] \\;=\\; \\mathrm{tr}(\\Sigma_2^{-1} \\Sigma_1) + (\\mu_1 - \\mu_2)^\\top \\Sigma_2^{-1}(\\mu_1 - \\mu_2)$$
      <p>Combining and pulling out $\\tfrac{1}{2}$ gives the boxed formula. $\\blacksquare$</p>
    `)}

    <div class="prose">
      <div id="kl-mvn-diag"></div>
      <h3>Diagonal covariances — the VAE regularizer</h3>
      <p>If $\\Sigma_1 = \\mathrm{diag}(\\sigma_{1,1}^2, \\ldots, \\sigma_{1,d}^2)$ and
      $\\Sigma_2 = \\mathrm{diag}(\\sigma_{2,1}^2, \\ldots, \\sigma_{2,d}^2)$, the formula
      simplifies to a sum of $d$ univariate KLs:</p>
      $$D_{\\mathrm{KL}}\\big(\\mathcal{N}(\\mu_1, \\Sigma_1) \\,\\big\\|\\, \\mathcal{N}(\\mu_2, \\Sigma_2)\\big)
        \\;=\\; \\tfrac{1}{2}\\sum_{i=1}^{d}\\!\\left[\\log \\frac{\\sigma_{2,i}^2}{\\sigma_{1,i}^2}
        \\;-\\; 1 \\;+\\; \\frac{\\sigma_{1,i}^2 + (\\mu_{1,i} - \\mu_{2,i})^2}{\\sigma_{2,i}^2}\\right]$$
      <p><strong>VAE special case</strong>: $q = \\mathcal{N}(\\mu, \\mathrm{diag}(\\sigma^2))$,
      $p = \\mathcal{N}(0, I)$:</p>
      $$\\boxed{\\;\\; D_{\\mathrm{KL}}\\big(q \\,\\|\\, \\mathcal{N}(0, I)\\big)
        \\;=\\; \\tfrac{1}{2}\\sum_{i=1}^{d}\\left[\\sigma_i^2 + \\mu_i^2 - 1 - \\log \\sigma_i^2\\right] \\;\\;}$$
      <p>Memorize this. Every VAE implementation has it as a one-line contribution to the loss.</p>

      <h3>Shared covariance</h3>
      <p>If $\\Sigma_1 = \\Sigma_2 = \\Sigma$, the log-det, trace, and $-d$ terms collapse:</p>
      $$D_{\\mathrm{KL}}\\big(\\mathcal{N}(\\mu_1, \\Sigma) \\,\\big\\|\\, \\mathcal{N}(\\mu_2, \\Sigma)\\big)
        \\;=\\; \\tfrac{1}{2}(\\mu_1 - \\mu_2)^\\top \\Sigma^{-1} (\\mu_1 - \\mu_2)$$
      <p>Just half the squared Mahalanobis distance between means. DDPM uses this: the
      per-timestep KL between two Gaussians sharing the same forward-process variance
      reduces to this clean form.</p>

      <h3>Worked numerical examples</h3>
      <table class="data-table">
        <thead>
          <tr>
            <th>$p$</th><th>$q$</th><th>$D_{\\mathrm{KL}}(p \\| q)$</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>$\\mathcal{N}([1, 0, {-1}], \\mathrm{diag}(1,2,1))$</td>
            <td>$\\mathcal{N}(0, I)$</td>
            <td>$1.1534$</td>
          </tr>
          <tr>
            <td>$\\mathcal{N}([0.5, {-0.2}], \\mathrm{diag}(e^{0.2}, e^{-0.6}))$</td>
            <td>$\\mathcal{N}(0, I)$</td>
            <td>$0.2301$</td>
          </tr>
          <tr>
            <td>$\\mathcal{N}([1, 1], \\bigl(\\begin{smallmatrix}1 & 0.5\\\\0.5 & 1\\end{smallmatrix}\\bigr))$</td>
            <td>$\\mathcal{N}([0, 0], 2I)$</td>
            <td>$0.8370$</td>
          </tr>
          <tr>
            <td>$\\mathcal{N}([0, 0], 2I)$</td>
            <td>$\\mathcal{N}([1, 1], \\bigl(\\begin{smallmatrix}1 & 0.5\\\\0.5 & 1\\end{smallmatrix}\\bigr))$</td>
            <td>$1.4963$</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div id="viz-kl-mvn-explorer" class="viz-wide"></div>

    ${crosslinkForward({
      toLesson: 'vae',
      toAnchor: 'vae-objective',
      body: `<p>The regularizer term in the VAE loss is this KL with
        $q = \\mathcal{N}(\\mu_\\phi(x), \\mathrm{diag}(\\sigma_\\phi^2(x)))$
        and $p = \\mathcal{N}(0, I)$ — the diagonal special case above.</p>`,
    })}

    ${crosslinkForward({
      toLesson: 'ddpm',
      body: `<p>Every per-timestep loss in DDPM is this KL with shared covariance.
        The diffusion chain ensures both distributions have the same variance schedule,
        leaving only the Mahalanobis term.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountKLMVNExplorer(sec.querySelector('#viz-kl-mvn-explorer') as HTMLElement);
}
