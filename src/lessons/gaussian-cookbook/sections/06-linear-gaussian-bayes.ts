import { renderMath, proofToggle, crosslinkSidebar } from '@shared/ui';
import { mountLinearGaussBayes } from '../viz/linear-gauss-bayes';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-6';
  sec.className = 'section';
  sec.dataset.anchor = 'linear-gauss-bayes';
  sec.innerHTML = `
    <div id="linear-gauss-bayes"></div>
    <div class="section-label">§6</div>
    <h2>The Linear-Gaussian Bayesian Update</h2>

    <div class="prose">
      <p>Given:</p>
      <ul>
        <li>Prior: $z \\sim \\mathcal{N}(\\mu_0, \\Sigma_0)$</li>
        <li>Likelihood: $x \\mid z \\sim \\mathcal{N}(A z + b, \\Sigma_n)$</li>
      </ul>
      <p>The posterior is Gaussian:</p>
      $$\\boxed{\\;\\; z \\mid x \\;\\sim\\; \\mathcal{N}(\\mu_{\\text{post}}, \\Sigma_{\\text{post}}) \\;\\;}$$
      <p>with</p>
      $$\\Sigma_{\\text{post}} \\;=\\; \\big(\\Sigma_0^{-1} + A^\\top \\Sigma_n^{-1} A\\big)^{-1}$$
      $$\\mu_{\\text{post}} \\;=\\; \\Sigma_{\\text{post}}\\!\\left[\\Sigma_0^{-1} \\mu_0 + A^\\top \\Sigma_n^{-1} (x - b)\\right]$$

      <h3>Reading the formulas</h3>
      <p><strong>Posterior precision</strong> = prior precision + likelihood precision:</p>
      $$\\Sigma_{\\text{post}}^{-1} \\;=\\; \\Sigma_0^{-1} \\;+\\; A^\\top \\Sigma_n^{-1} A$$
      <p>Precisions add. The observation contributes an "evidence precision"
      $A^\\top \\Sigma_n^{-1} A$ proportional to how informative the linear map $A$ is.</p>
      <p><strong>Posterior mean</strong> = precision-weighted combination of prior mean
      and observation:</p>
      $$\\Sigma_{\\text{post}}^{-1} \\mu_{\\text{post}} \\;=\\; \\Sigma_0^{-1} \\mu_0 \\;+\\; A^\\top \\Sigma_n^{-1} (x - b)$$
      <p>Stronger likelihood (smaller $\\Sigma_n$) pulls $\\mu_{\\text{post}}$ toward
      $A^{-1}(x - b)$. Stronger prior (smaller $\\Sigma_0$) pulls $\\mu_{\\text{post}}$
      toward $\\mu_0$.</p>
    </div>

    ${proofToggle('Derivation (two routes)', `
      <p><strong>Route 1 (complete the square):</strong></p>
      $$\\log p(z \\mid x) \\;=\\; \\log p(x \\mid z) + \\log p(z) + \\text{const}$$
      $$\\propto -\\tfrac{1}{2}(x - Az - b)^\\top \\Sigma_n^{-1}(x - Az - b)
        \\;-\\; \\tfrac{1}{2}(z - \\mu_0)^\\top \\Sigma_0^{-1}(z - \\mu_0)$$
      <p>Expand and collect terms quadratic in $z$. The coefficient of $-\\tfrac{1}{2}z^\\top(\\cdot)z$
      is $A^\\top \\Sigma_n^{-1} A + \\Sigma_0^{-1}$, giving $\\Sigma_{\\text{post}}^{-1}$.
      The linear coefficient gives $\\Sigma_{\\text{post}}^{-1}\\mu_{\\text{post}}$. $\\blacksquare$</p>

      <p><strong>Route 2 (apply §5 conditioning):</strong>
      $(z, x)$ is jointly Gaussian because $x = Az + b + \\varepsilon_n$ is an affine
      transformation of a Gaussian plus Gaussian noise. Write the joint covariance, then
      apply the conditioning formula from §5. After block-matrix manipulation, the same
      answer falls out. $\\blacksquare$</p>
    `)}

    <div class="prose">
      <div class="worked-example">
        <div class="worked-example-title">Worked numerical example</div>
        <p>Take a 2D prior and a noisy linear observation:</p>
        <ul>
          <li>Prior: $z \\sim \\mathcal{N}(0, I_2)$</li>
          <li>Likelihood: $x \\mid z \\sim \\mathcal{N}\\!\\left(\\begin{pmatrix}1 & 0.5 \\\\ 0.3 & 1\\end{pmatrix} z, \\;\\; 0.1 \\, I_2\\right)$</li>
          <li>Observe: $x = (1.5, 0.8)$</li>
        </ul>
        <p>Computing:</p>
        $$A^\\top \\Sigma_n^{-1} A \\;=\\; 10 \\begin{pmatrix}1 & 0.3 \\\\ 0.5 & 1\\end{pmatrix}\\begin{pmatrix}1 & 0.5 \\\\ 0.3 & 1\\end{pmatrix}
          \\;=\\; 10 \\begin{pmatrix}1.09 & 0.8 \\\\ 0.8 & 1.25\\end{pmatrix}$$
        $$\\Sigma_{\\text{post}}^{-1} \\;=\\; I + 10\\begin{pmatrix}1.09 & 0.8 \\\\ 0.8 & 1.25\\end{pmatrix}
          \\;=\\; \\begin{pmatrix}11.9 & 8 \\\\ 8 & 13.5\\end{pmatrix}$$
        $$\\Sigma_{\\text{post}} \\;\\approx\\; \\begin{pmatrix}0.1397 & -0.0828 \\\\ -0.0828 & 0.1231\\end{pmatrix},
          \\qquad \\mu_{\\text{post}} \\;\\approx\\; (1.147, \\; 0.468)$$
      </div>
    </div>

    <div id="viz-linear-gauss-bayes" class="viz-wide"></div>

    ${crosslinkSidebar({
      toLesson: 'langevin',
      body: `<p>Kalman filtering (not built as a standalone lesson) is this identity
        applied repeatedly as observations arrive. Gaussian processes use the conditioning
        formula (§5) to update beliefs given data.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountLinearGaussBayes(sec.querySelector('#viz-linear-gauss-bayes') as HTMLElement);
}
