import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-2';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'score-definition');
  sec.innerHTML = `
    <div class="section-label">§2</div>
    <h2>The Score Function</h2>
    <div class="prose">
      <p>The <strong>score function</strong> of a density $p$ is its gradient on a log scale:</p>

      <div class="formula-box">$$s(x) := \\nabla_x \\log p(x)$$</div>

      <p>Beware: this name collides with the statistics convention where "score" means
      $\\nabla_\\theta \\log p(x; \\theta)$ — the gradient with respect to <em>parameters</em>.
      In score-based modeling, we always mean the gradient with respect to the <em>input</em>
      $x$, holding the distribution fixed.</p>

      <h3>Geometric meaning</h3>
      <p>$s(x)$ is a vector field on $\\mathbb{R}^d$. At every point $x$ it outputs
      a $d$-dimensional vector pointing in the direction of steepest ascent of $\\log p$ —
      toward higher density. Three immediate observations:</p>
      <ul>
        <li><strong>At modes</strong> (local maxima of $p$), $s(x) = 0$. The score vanishes
        wherever the density is locally peaked.</li>
        <li><strong>Far from data</strong>, where $p$ is small, the score points toward
        regions where $p$ is larger. The score field "points home" no matter where you are.</li>
        <li><strong>Magnitude</strong> of $\\|s(x)\\|$ measures how rapidly $\\log p$ is
        changing. Sharp peaks → large score nearby; broad plateaus → small score.</li>
      </ul>

      <h3>Why the normalization disappears</h3>
      <p>Write any density as $p(x) = \\tilde{p}(x) / Z$ where $\\tilde{p}$ is any
      non-negative function and $Z = \\int \\tilde{p}(x)\\,dx$. Then:</p>
      $$\\log p(x) = \\log \\tilde{p}(x) - \\log Z$$
      <p>Taking the gradient with respect to $x$:</p>
      $$s(x) = \\nabla_x \\log \\tilde{p}(x) - \\underbrace{\\nabla_x \\log Z}_{= 0} = \\nabla_x \\log \\tilde{p}(x)$$
      <p><strong>The score is invariant to normalization.</strong> We can specify an
      unnormalized energy $\\tilde{p}(x) = e^{-U(x)}$ and the score is just
      $-\\nabla U(x)$ — no integration over $\\mathbb{R}^d$ required.</p>

      <h3>Three worked examples</h3>
      <p><strong>Standard normal.</strong> If $p(x) = \\mathcal{N}(x;\\, 0, I)$, then
      $\\log p(x) = -\\tfrac{1}{2}\\|x\\|^2 + \\text{const}$, so
      $s(x) = -x$.
      A linear vector field pointing radially toward the origin.</p>

      <p><strong>General Gaussian.</strong> For $p(x) = \\mathcal{N}(x;\\, \\mu, \\Sigma)$:</p>
      <div class="formula-box">$$s(x) = -\\Sigma^{-1}(x - \\mu)$$</div>
      <p>At $\\mu = 0$, $\\Sigma = \\begin{pmatrix}1 & 0.5 \\\\ 0.5 & 1\\end{pmatrix}$,
      $x = (1, 0)$:</p>
      $$s(x) = -\\frac{1}{0.75}\\begin{pmatrix}1 & -0.5 \\\\ -0.5 & 1\\end{pmatrix}
      \\begin{pmatrix}1 \\\\ 0\\end{pmatrix} = \\begin{pmatrix}-4/3 \\\\ 2/3\\end{pmatrix}$$

      <p><strong>Gaussian mixture.</strong> For $p(x) = \\sum_k \\pi_k \\mathcal{N}(x;\\, \\mu_k, \\Sigma_k)$:</p>
      $$s(x) = \\sum_k r_k(x) \\cdot \\big[-\\Sigma_k^{-1}(x - \\mu_k)\\big]$$
      <p>where $r_k(x) = \\pi_k \\mathcal{N}(x;\\, \\mu_k, \\Sigma_k) / p(x)$ is the
      responsibility of component $k$ — the same quantity as the E-step in EM.
      The mixture score is the responsibility-weighted average of the component scores.
      <strong>Saddles emerge</strong> at points equidistant between modes, where
      competing components pull in opposite directions.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'em',
      toAnchor: 'q-function',
      toAnchorLabel: '4 — the E-step',
      body: `The responsibilities $r_k(x)$ that weight the mixture-score components
             are the same posterior-responsibility quantities that EM computes in its
             E-step. Different framework, same statistical object.`,
    })}

    <div id="viz-score-field-explorer" class="viz-placeholder">
      <p class="viz-placeholder__label">Visualization: Score Field Explorer</p>
    </div>
    <div id="viz-normalization-irrelevance" class="viz-placeholder">
      <p class="viz-placeholder__label">Visualization: Normalization Irrelevance</p>
    </div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
