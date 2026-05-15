import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-2';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'forward-process');
  sec.innerHTML = `
    <div class="section-label">§2</div>
    <h2 id="forward-process">The Forward Process: Destroying Signal</h2>
    <div class="prose">
      <p>The <strong>forward process</strong> (also called the
      <strong>diffusion process</strong>) is a fixed Markov chain that gradually
      adds Gaussian noise to data until nothing recognizable remains. Define it
      by:</p>

      <div class="formula-box">$$q(x_t \\mid x_{t-1}) \\;=\\; \\mathcal{N}\\!\\big(x_t; \\;\\sqrt{1 - \\beta_t} \\, x_{t-1}, \\; \\beta_t I\\big)$$</div>

      <p>where $\\beta_1, \\beta_2, \\ldots, \\beta_T \\in (0, 1)$ is a small
      sequence of variances (the <strong>noise schedule</strong>). At each step:
      shrink the previous sample by $\\sqrt{1 - \\beta_t}$ and add Gaussian noise
      with variance $\\beta_t$. The full forward distribution is:</p>

      $$q(x_{1:T} \\mid x_0) \\;=\\; \\prod_{t=1}^{T} q(x_t \\mid x_{t-1})$$

      <h3>Why the $\\sqrt{1 - \\beta_t}$ scaling</h3>
      <p>Without the scaling factor, variance would <em>accumulate</em>: each step
      adds $\\beta_t$ of variance on top of whatever variance the previous
      $x_{t-1}$ had. After many steps, $\\mathrm{Var}(x_t)$ would grow without
      bound.</p>

      <p>With the scaling: if $\\mathrm{Var}(x_{t-1}) = 1$, then $\\mathrm{Var}(x_t)
      = (1 - \\beta_t) \\cdot 1 + \\beta_t = 1$. <strong>Variance is preserved.</strong>
      A unit-variance signal $x_0$ stays approximately unit-variance throughout the
      chain. This is essential because the reverse process eventually has to start
      from $\\mathcal{N}(0, I)$ — a unit-variance distribution — so the chain
      endpoints have to match.</p>

      <p>The paper's exact schedule: $T = 1000$, $\\beta_t$ linear from
      $\\beta_1 = 10^{-4}$ to $\\beta_T = 0.02$. Small at the start (so early steps
      preserve fine detail) and slightly larger at the end (so the chain reaches
      near-Gaussian noise by $T$).</p>

      <h3 id="closed-form-marginal">The closed-form jump</h3>
      <p>The remarkable property of the forward process is that <strong>we can
      sample $x_t$ from $x_0$ directly, without iterating through the intermediate
      steps.</strong> Define:</p>

      $$\\alpha_t \\;:=\\; 1 - \\beta_t, \\qquad \\bar\\alpha_t \\;:=\\; \\prod_{s=1}^{t} \\alpha_s$$

      <p>Then:</p>

      <div class="formula-box">$$q(x_t \\mid x_0) \\;=\\; \\mathcal{N}\\!\\big(x_t; \\; \\sqrt{\\bar\\alpha_t} \\, x_0, \\; (1 - \\bar\\alpha_t) I\\big)$$</div>

      <p>One Gaussian, available in closed form. This is the single most important
      property of the forward process — without it, training would require an
      $O(T)$ inner loop per data point.</p>

      <p><strong>Derivation by induction.</strong> The base case $t = 1$ is the
      definition. For the inductive step, assume
      $x_{t-1} = \\sqrt{\\bar\\alpha_{t-1}} \\, x_0 + \\sqrt{1 - \\bar\\alpha_{t-1}} \\, \\tilde\\epsilon$
      for $\\tilde\\epsilon \\sim \\mathcal{N}(0, I)$. Then</p>

      $$x_t \\;=\\; \\sqrt{1 - \\beta_t} \\, x_{t-1} + \\sqrt{\\beta_t} \\, \\epsilon$$

      <p>Combining the two independent Gaussian contributions:
      $\\alpha_t(1 - \\bar\\alpha_{t-1}) + \\beta_t = 1 - \\bar\\alpha_t$. So
      $x_t = \\sqrt{\\bar\\alpha_t} \\, x_0 + \\sqrt{1 - \\bar\\alpha_t} \\, \\epsilon'$
      for some standard Gaussian $\\epsilon'$. $\\blacksquare$</p>

      <h3>Reading the marginal</h3>
      $$x_t \\;=\\; \\sqrt{\\bar\\alpha_t} \\, x_0 \\;+\\; \\sqrt{1 - \\bar\\alpha_t} \\, \\epsilon, \\qquad \\epsilon \\sim \\mathcal{N}(0, I)$$

      <p>Two terms: $\\sqrt{\\bar\\alpha_t} \\, x_0$ is the <strong>signal</strong>
      — a shrinking copy of the original data; $\\sqrt{1 - \\bar\\alpha_t} \\,
      \\epsilon$ is the <strong>noise</strong> — a growing standard Gaussian
      contribution. The signal-to-noise ratio is
      $\\bar\\alpha_t / (1 - \\bar\\alpha_t)$, which <strong>decreases monotonically
      with $t$</strong>. At $t = 0$: pure signal. At $t = T$: pure noise.</p>

      <h3>Worked numerical values</h3>
      <p>For the paper's schedule ($T = 1000$, $\\beta_t$ linear from $10^{-4}$ to
      $0.02$):</p>
      <table class="num-table">
        <thead><tr><th>$t$</th><th>$\\beta_t$</th><th>$\\bar\\alpha_t$</th><th>$\\sqrt{\\bar\\alpha_t}$</th><th>$\\sqrt{1 - \\bar\\alpha_t}$</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>$10^{-4}$</td><td>0.9998</td><td>0.9999</td><td>0.0148</td></tr>
          <tr><td>100</td><td>0.0021</td><td>0.895</td><td>0.946</td><td>0.324</td></tr>
          <tr><td>250</td><td>0.0051</td><td>0.521</td><td>0.722</td><td>0.692</td></tr>
          <tr><td>500</td><td>0.0101</td><td>0.0778</td><td>0.279</td><td>0.960</td></tr>
          <tr><td>750</td><td>0.0150</td><td>0.0033</td><td>0.057</td><td>0.998</td></tr>
          <tr><td>999</td><td>0.0200</td><td>$4 \\cdot 10^{-5}$</td><td>0.006</td><td>1.000</td></tr>
        </tbody>
      </table>
      <p>Note: by $t = 500$, <strong>less than 8% of the original signal variance
      survives.</strong> By $t = 1000$, less than $4 \\cdot 10^{-3}$% — the data is
      essentially pure standard normal noise.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'reparam-matrix',
      toAnchorLabel: '4 — reparameterization',
      body: `The closed-form $x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1 - \\bar\\alpha_t} \\epsilon$
             is the reparameterization trick: $\\sqrt{\\bar\\alpha_t} x_0$ acts as the mean,
             $\\sqrt{1 - \\bar\\alpha_t}$ as the scale, $\\epsilon$ as the noise. Two
             affine-transformed Gaussians compose into one Gaussian with the matching mean
             and variance.`,
    })}

    <div id="viz-forward-chain" style="margin-top:1.5rem;"></div>
    <div id="viz-closed-form-jump" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
