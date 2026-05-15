import { renderMath } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-6';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'langevin');
  sec.innerHTML = `
    <div class="section-label">§6</div>
    <h2>Langevin Dynamics for Sampling</h2>
    <div class="prose">
      <p>We now have a trained score model $s_\\theta(x) \\approx \\nabla \\log p(x)$.
      The question: how do we use it to <strong>sample</strong> from $p$?</p>

      <p>The answer is <strong>Langevin dynamics</strong>, a stochastic process that uses
      the score as a drift term and Gaussian noise to explore.</p>

      <h3>The continuous-time form</h3>
      <p>Langevin dynamics is the SDE:</p>
      <div class="formula-box">
        $$dx_t = \\nabla_x \\log p(x_t)\\, dt + \\sqrt{2}\\, dW_t$$
      </div>
      <p>where $W_t$ is a standard Brownian motion. Under mild regularity conditions,
      <strong>the stationary distribution of this SDE is exactly $p$</strong> — start any
      trajectory anywhere, let it run long enough, and the resulting distribution
      converges to $p$.</p>

      <p>Two competing forces:</p>
      <ul>
        <li><strong>Drift:</strong> $\\nabla \\log p$ pulls $x_t$ toward higher density.
        Without noise, this would be gradient ascent — converging to a mode.</li>
        <li><strong>Diffusion:</strong> $\\sqrt{2}\\,dW_t$ adds Gaussian noise. Prevents
        collapse to modes; encourages exploration.</li>
      </ul>
      <p>The balance is set so that detailed balance holds with respect to $p$ — the
      Fokker–Planck equation has $p$ as its stationary solution.</p>

      <h3>The discrete-time algorithm</h3>
      <p>Discretize the SDE with step size $\\eta$:</p>
      <div class="formula-box">
        $$x_{t+1} = x_t + \\eta\\, \\nabla \\log p(x_t) + \\sqrt{2\\eta}\\, \\varepsilon_t,
        \\qquad \\varepsilon_t \\sim \\mathcal{N}(0, I)$$
      </div>
      <p>In practice we replace $\\nabla \\log p$ with $s_\\theta$:</p>
      $$x_{t+1} = x_t + \\eta\\, s_\\theta(x_t) + \\sqrt{2\\eta}\\, \\varepsilon_t$$

      <h3>Worked example: one step on $\\mathcal{N}(0, I)$</h3>
      <p>Start at $x_0 = (1, 1)$. True score: $s(x_0) = -(1,1)$.
      Step size $\\eta = 0.1$, noise $\\varepsilon = (0.2, -0.3)$:</p>
      $$x_1 = (1,1) + 0.1 \\cdot (-1,-1) + \\sqrt{0.2} \\cdot (0.2, -0.3)$$
      $$= (1,1) + (-0.1, -0.1) + (0.0894,\\, -0.1342) \\approx (0.9894,\\, 0.7658)$$
      <p>The drift pulled $x$ toward the origin; noise nudged it slightly off-axis.</p>

      <h3>Why naive Langevin fails on real data</h3>
      <p>Plain Langevin with a learned $s_\\theta$ has two failure modes:</p>
      <ol>
        <li><strong>The score is unreliable off the data manifold.</strong> Real data
        lives on a low-dimensional manifold in high-dim space. The score model is trained
        on $p_{\\mathrm{data}}$ — it never sees points far from the manifold, so $s_\\theta$
        there is meaningless. A particle initialized in this no-signal region wanders
        randomly.</li>
        <li><strong>Multimodal mixing is slow.</strong> Even when $s_\\theta$ is accurate,
        crossing between modes requires the noise to randomly drive $x_t$ over a
        low-density barrier. With small $\\eta$, this takes exponentially many steps.</li>
      </ol>
      <p>Both problems point in the same direction: <strong>train the score at multiple
      noise levels.</strong> At high noise, the score field is smooth and globally
      informative; at low noise, sharp and locally precise. <strong>Annealed Langevin
      sampling</strong> uses both — §7.</p>
    </div>

    <div id="viz-langevin-sampler" class="viz-placeholder">
      <p class="viz-placeholder__label">Visualization: Langevin Sampler (analytical score)</p>
    </div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
