import { renderMath, proofToggle, crosslinkForward } from '@shared/ui';
import { mountConditioningSlice } from '../viz/conditioning-slice';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-5';
  sec.className = 'section';
  sec.dataset.anchor = 'conditioning';
  sec.innerHTML = `
    <div id="conditioning"></div>
    <div class="section-label">§5</div>
    <h2>Conditioning and Marginalization</h2>

    <div class="prose">
      <p>Let $(X, Y) \\sim \\mathcal{N}\\!\\left(\\begin{pmatrix}\\mu_X \\\\ \\mu_Y\\end{pmatrix},
      \\begin{pmatrix}\\Sigma_{XX} & \\Sigma_{XY} \\\\ \\Sigma_{YX} & \\Sigma_{YY}\\end{pmatrix}\\right)$
      be jointly Gaussian. Then:</p>

      $$\\boxed{\\;\\; X \\;\\sim\\; \\mathcal{N}(\\mu_X, \\Sigma_{XX}) \\;\\;}$$

      $$\\boxed{\\;\\; Y \\mid X = x \\;\\sim\\; \\mathcal{N}\\!\\big(\\mu_Y + \\Sigma_{YX} \\Sigma_{XX}^{-1} (x - \\mu_X), \\;\\; \\Sigma_{YY} - \\Sigma_{YX} \\Sigma_{XX}^{-1} \\Sigma_{XY}\\big) \\;\\;}$$

      <p>A jointly Gaussian distribution stays Gaussian under both <strong>marginalization</strong>
      (drop $Y$, read off the relevant block) and <strong>conditioning</strong>
      (fix $X$, get the formula above).</p>

      <h3>Reading the conditional mean</h3>
      $$\\mu_{Y \\mid X = x} \\;=\\; \\mu_Y + \\Sigma_{YX} \\Sigma_{XX}^{-1}(x - \\mu_X)$$
      <p>Start at $\\mu_Y$ and shift by an amount proportional to how much $x$ deviates
      from $\\mu_X$. The "amount" is governed by $\\Sigma_{YX} \\Sigma_{XX}^{-1}$ — the
      <strong>regression coefficient</strong> of $Y$ on $X$. When $\\Sigma_{YX} = 0$,
      knowing $X$ tells you nothing; when they're highly correlated, the shift is large.</p>

      <h3>Reading the conditional covariance</h3>
      $$\\Sigma_{Y \\mid X} \\;=\\; \\Sigma_{YY} - \\Sigma_{YX} \\Sigma_{XX}^{-1} \\Sigma_{XY}$$
      <p>Start at the marginal covariance of $Y$ and <strong>subtract</strong> the
      variance "explained" by $X$ (the Schur complement of $\\Sigma_{XX}$). Conditioning
      on $X$ never increases uncertainty about $Y$: the subtracted term is always
      positive semidefinite.</p>
    </div>

    ${proofToggle('Derivation (block matrix / Schur complement)', `
      <p>Strategy: complete the square in $y$ inside the joint density. Use the
      <strong>block matrix inversion</strong> formula. Write</p>
      $$\\Sigma^{-1} \\;=\\; \\begin{pmatrix} \\Sigma_{XX} & \\Sigma_{XY} \\\\ \\Sigma_{YX} & \\Sigma_{YY} \\end{pmatrix}^{\\!-1}$$
      <p>Standard result (Schur complement of $\\Sigma_{XX}$):</p>
      $$\\Sigma^{-1} \\;=\\; \\begin{pmatrix}
        \\Sigma_{XX}^{-1} + \\Sigma_{XX}^{-1}\\Sigma_{XY} S^{-1} \\Sigma_{YX}\\Sigma_{XX}^{-1}
          & -\\Sigma_{XX}^{-1}\\Sigma_{XY} S^{-1} \\\\
        -S^{-1} \\Sigma_{YX}\\Sigma_{XX}^{-1} & S^{-1}
      \\end{pmatrix}$$
      <p>where $S = \\Sigma_{YY} - \\Sigma_{YX} \\Sigma_{XX}^{-1}\\Sigma_{XY}$ is the
      <strong>Schur complement of $\\Sigma_{XX}$</strong> — exactly the conditional covariance.</p>
      <p>The joint log-density has $\\propto -\\tfrac{1}{2}\\begin{pmatrix}x - \\mu_X \\\\ y - \\mu_Y\\end{pmatrix}^\\top \\Sigma^{-1}\\begin{pmatrix}x - \\mu_X \\\\ y - \\mu_Y\\end{pmatrix}$ as the only term involving $y$. Expanding the $y$-dependent blocks:</p>
      $$\\propto -\\tfrac{1}{2}(y - \\mu_Y)^\\top S^{-1}(y - \\mu_Y)
        \\;+\\; (y - \\mu_Y)^\\top S^{-1} \\Sigma_{YX}\\Sigma_{XX}^{-1}(x - \\mu_X)
        \\;+\\; (\\text{const in } y)$$
      <p>Completing the square in $y$ around
      $\\mu_Y + \\Sigma_{YX}\\Sigma_{XX}^{-1}(x - \\mu_X)$ gives a Gaussian in $y$
      with covariance $S$ and that mean — matching the boxed identity. $\\blacksquare$</p>
    `)}

    <div class="prose">
      <div class="worked-example">
        <div class="worked-example-title">Worked numerical example</div>
        <p>Take $(X, Y) \\sim \\mathcal{N}\\!\\left(\\begin{pmatrix}0 \\\\ 0\\end{pmatrix},
        \\begin{pmatrix}1 & 0.7 \\\\ 0.7 & 1\\end{pmatrix}\\right)$ and observe $X = 1$. Then:</p>
        $$\\mu_{Y \\mid X = 1} \\;=\\; 0 + 0.7 \\cdot 1 \\cdot (1 - 0) \\;=\\; 0.7$$
        $$\\sigma_{Y \\mid X}^{\\,2} \\;=\\; 1 - 0.7 \\cdot 1 \\cdot 0.7 \\;=\\; 0.51$$
        <p>So $Y \\mid X = 1 \\sim \\mathcal{N}(0.7, 0.51)$. The mean is <strong>pulled
        toward</strong> $X$ (correlation 0.7 → 70% of the deviation passes through), and
        the conditional variance (0.51) is <strong>smaller</strong> than the marginal (1.0)
        because knowing $X$ has resolved some uncertainty about $Y$.</p>
      </div>
    </div>

    <div id="viz-conditioning-slice" class="viz-wide"></div>

    ${crosslinkForward({
      toLesson: 'ddpm',
      body: `<p>The analytical posterior $q(z_{t-1} \\mid z_t, x_0)$ that the score network
        targets is derived via this identity. The forward process $(z_0, z_1, \\ldots, z_T)$
        is jointly Gaussian; this formula gives the conditional in closed form.</p>`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountConditioningSlice(sec.querySelector('#viz-conditioning-slice') as HTMLElement);
}
