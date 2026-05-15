import { renderMath } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-6';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'eps-parameterization');
  sec.innerHTML = `
    <div class="section-label">§6</div>
    <h2 id="eps-parameterization">Parameterizing the Reverse Process</h2>
    <div class="prose">
      <p>The variational bound, post-decomposition, says: train $\\mu_\\theta(x_t, t)$
      to match $\\tilde\\mu_t(x_t, x_0)$ (with $x_0$ in the expectation). How should
      we parameterize $\\mu_\\theta$?</p>

      <p>Three natural choices, <strong>mathematically equivalent</strong>:</p>
      <ol>
        <li>Predict the posterior mean directly: $\\mu_\\theta(x_t, t) \\approx \\tilde\\mu_t$.</li>
        <li>Predict $x_0$ from $x_t$, then plug into $\\tilde\\mu_t$:
        $\\mu_\\theta(x_t, t) = \\tilde\\mu_t(x_t, x_0^{\\theta}(x_t))$.</li>
        <li>Predict the noise $\\epsilon$ from $x_t$, then back out $\\mu_\\theta$.</li>
      </ol>

      <p>All three lead to the same optimal $\\mu_\\theta^*$. They differ in
      <strong>what the network is asked to output</strong> — and this affects
      optimization dynamics significantly. The paper finds (and shows empirically
      in Table 2) that <strong>option 3 — predicting $\\epsilon$ — works best.</strong></p>

      <h3>Deriving the $\\epsilon$-prediction parameterization</h3>
      <p>Start from the marginal
      $x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1 - \\bar\\alpha_t} \\epsilon$.
      Solve for $x_0$:</p>

      $$x_0 \\;=\\; \\frac{1}{\\sqrt{\\bar\\alpha_t}}\\!\\left(x_t - \\sqrt{1 - \\bar\\alpha_t} \\, \\epsilon\\right)$$

      <p>Substitute into the posterior mean $\\tilde\\mu_t(x_t, x_0)$ from §5 and
      simplify. After algebra (the coefficient of $x_t$ collapses to
      $1/\\sqrt{\\alpha_t}$ via $\\alpha_t + \\beta_t = 1$ and
      $\\bar\\alpha_t = \\alpha_t \\bar\\alpha_{t-1}$):</p>

      <div class="formula-box">$$\\tilde\\mu_t(x_t, x_0(\\epsilon, x_t)) \\;=\\; \\frac{1}{\\sqrt{\\alpha_t}}\\!\\left(x_t \\;-\\; \\frac{\\beta_t}{\\sqrt{1 - \\bar\\alpha_t}} \\epsilon\\right)$$</div>

      <p><strong>The posterior mean, in terms of $x_t$ and the noise $\\epsilon$
      that produced it, has a remarkably clean form.</strong> Two terms:
      $x_t / \\sqrt{\\alpha_t}$ (current sample, slightly amplified) and
      $-\\beta_t \\epsilon / (\\sqrt{\\alpha_t} \\sqrt{1 - \\bar\\alpha_t})$
      (a correction proportional to the noise).</p>

      <p>The reverse-process mean parameterization is then:</p>

      <div class="formula-box">$$\\mu_\\theta(x_t, t) \\;=\\; \\frac{1}{\\sqrt{\\alpha_t}}\\!\\left(x_t \\;-\\; \\frac{\\beta_t}{\\sqrt{1 - \\bar\\alpha_t}} \\epsilon_\\theta(x_t, t)\\right)$$</div>

      <p>The network $\\epsilon_\\theta(x_t, t)$ outputs a vector of the same shape
      as $x_t$, interpreted as the model's estimate of "what noise was added to
      produce $x_t$."</p>

      <h3 id="L-simple">What the loss becomes</h3>
      <p>Substitute the $\\epsilon$-parameterized $\\mu_\\theta$ into
      $L_{t-1} = \\frac{1}{2\\sigma_t^2} \\|\\tilde\\mu_t - \\mu_\\theta\\|^2$.
      The $1/\\sqrt{\\alpha_t}$ and $\\beta_t/\\sqrt{1 - \\bar\\alpha_t}$ factors
      come out as a per-$t$ constant:</p>

      $$L_{t-1} \\;=\\; \\frac{\\beta_t^2}{2 \\sigma_t^2 \\alpha_t (1 - \\bar\\alpha_t)} \\, \\mathbb{E}_{x_0, \\epsilon}\\!\\left[\\big\\|\\epsilon - \\epsilon_\\theta\\!\\big(\\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1 - \\bar\\alpha_t} \\epsilon, \\;t\\big)\\big\\|^2\\right]$$

      <p>(This is Equation 12 of the paper.) <strong>The model is asked to predict
      $\\epsilon$, the noise that was added.</strong> The loss is MSE on noise
      prediction, weighted by a per-timestep factor.</p>

      <p>If we set $\\sigma_t^2 = \\beta_t$ and drop all per-$t$ weights:</p>

      <div class="formula-box">$$L_{\\text{simple}}(\\theta) \\;=\\; \\mathbb{E}_{t, x_0, \\epsilon}\\!\\left[\\big\\|\\epsilon - \\epsilon_\\theta\\!\\big(\\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1 - \\bar\\alpha_t} \\epsilon, \\;t\\big)\\big\\|^2\\right]$$</div>

      <p>(Equation 14 of the paper.) The whole loss collapses to: <strong>predict
      the noise from the noisy sample.</strong> Eight lines of code.</p>

      <h3>Why $\\epsilon$-prediction works better</h3>
      <p>The paper's empirical result (Table 2): predicting $\\epsilon$ with
      $L_{\\text{simple}}$ gives FID 3.17 on CIFAR10; predicting $\\tilde\\mu$ with
      $L_{\\text{simple}}$ doesn't converge.</p>

      <p>Three plausible reasons:</p>
      <ol>
        <li><strong>Scale stability.</strong> $\\epsilon$ has unit variance for all
        $t$ (it's standard Gaussian by construction). $\\tilde\\mu_t$ has variance
        that scales with $\\bar\\alpha_t$ — small at large $t$. A unit-variance
        target lets the same network width and learning rate work across all $t$.</li>
        <li><strong>The down-weighting of small-$t$ terms by dropping the weight.</strong>
        The paper-derived weight
        $\\beta_t^2 / (2 \\sigma_t^2 \\alpha_t (1 - \\bar\\alpha_t))$ is <strong>larger
        at small $t$</strong> (where the loss is "easier" — almost-clean data).
        $L_{\\text{simple}}$ flattens this. The intuition: focus capacity on harder
        denoising at large $t$, where the model actually has to <em>generate</em>
        structure.</li>
        <li><strong>Implicit connection to score matching.</strong> With
        $\\epsilon$-prediction, the loss is literally denoising score matching
        (next section). The optimization is well-conditioned because the target
        geometry is the score field.</li>
      </ol>

      <h3>Numerical example</h3>
      <p>At $t = 500$ in the paper's schedule: $\\beta_t = 0.0101$,
      $\\alpha_t = 0.9899$, $\\bar\\alpha_t = 0.078$. With $\\sigma_t^2 = \\beta_t$:</p>

      $$\\text{weight}_{500} \\;=\\; \\frac{(0.0101)^2}{2 \\cdot 0.0101 \\cdot 0.9899 \\cdot 0.922} \\;\\approx\\; 0.0055$$

      <p>Compare to $t = 1$:</p>

      $$\\text{weight}_{1} \\;=\\; \\frac{(10^{-4})^2}{2 \\cdot 10^{-4} \\cdot 0.9999 \\cdot 2 \\cdot 10^{-4}} \\;\\approx\\; 0.25$$

      <p>The weight at $t = 1$ is <strong>~45× the weight at $t = 500$</strong>.
      Under the full bound, the model is heavily pushed to be accurate on
      easy-denoising tasks. $L_{\\text{simple}}$ undoes this.</p>
    </div>

    <div id="viz-parameterization-comparison" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
