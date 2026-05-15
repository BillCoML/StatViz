import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-9';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'sampling-algorithm');
  sec.innerHTML = `
    <div class="section-label">§9</div>
    <h2 id="sampling-algorithm">The Sampling Algorithm</h2>
    <div class="prose">
      <p>The paper's Algorithm 2 is six lines:</p>

      <pre class="algorithm-block">Algorithm 2 — Sampling
1: x_T ~ N(0, I)
2: for t = T, ..., 1 do
3:    z ~ N(0, I) if t > 1, else z = 0
4:    x_{t-1} = (1 / sqrt(alpha_t)) (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
5: end for
6: return x_0</pre>

      <p>Sample pure noise. Then for each of $T$ steps, apply the reverse update:
      compute the conditional mean, add a fresh dose of noise, repeat. After $T$
      steps, you have a sample from $p_\\theta(x_0)$.</p>

      <h3>Reading the update rule</h3>
      $$x_{t-1} \\;=\\; \\underbrace{\\frac{1}{\\sqrt{\\alpha_t}}\\!\\left(x_t \\;-\\; \\frac{1 - \\alpha_t}{\\sqrt{1 - \\bar\\alpha_t}} \\, \\epsilon_\\theta(x_t, t)\\right)}_{\\mu_\\theta(x_t, t)} \\;+\\; \\sigma_t \\, z$$

      <p>The first part is the $\\epsilon$-parameterized $\\mu_\\theta(x_t, t)$
      from §6. The second part — adding $\\sigma_t z$ — is <strong>Langevin noise
      injection</strong> from Score Matching §6.</p>

      <p>Why add noise during sampling? Two reasons:</p>
      <ol>
        <li><strong>It's variational inference.</strong> The reverse process
        $p_\\theta(x_{t-1} \\mid x_t)$ is parameterized as a Gaussian; sampling
        means drawing from that Gaussian (mean plus noise). Setting $z = 0$ would
        collapse every sample to the mean — not actually sampling.</li>
        <li><strong>It's Langevin sampling.</strong> Pure gradient descent on
        $\\log p$ collapses to modes. The noise term is what lets the sampler
        explore and approximate the full distribution.</li>
      </ol>

      <p>The two reasons are the same reason in different framings. (Of course
      they are — that's the whole §7 punchline.)</p>

      <h3>The final step ($t = 1$)</h3>
      <p>At $t = 1$, we set $z = 0$. The final sample is $\\mu_\\theta(x_1, 1)$ —
      deterministic given $x_1$. Why? Because $x_0$ should be a clean data point,
      not have additional Gaussian noise added at the end. The "Langevin
      temperature" cools to zero at the final step.</p>

      <h3>Tracing one sampling step numerically</h3>
      <p>Suppose at $t = 500$ we have $x_t = (0.567, -0.812)$ and the model
      outputs $\\epsilon_\\theta(x_t, 500) = (0.3, -0.7)$ (a "perfect" prediction
      for the §8 trace). Fresh noise $z = (0.1, -0.2)$,
      $\\sigma_t = \\sqrt{\\beta_{500}} \\approx 0.1003$:</p>
      <ul>
        <li>$\\mu_\\theta = \\frac{1}{\\sqrt{0.9899}}\\!\\left((0.567, -0.812) - \\frac{0.0101}{0.9603}(0.3, -0.7)\\right)$</li>
        <li>$\\quad\\;\\;\\, \\approx (0.5667, -0.8084)$</li>
        <li>$\\sigma_t z = (0.01003, -0.02006)$</li>
        <li>$x_{t-1} \\approx (0.577, -0.828)$</li>
      </ul>

      <p>The sample has moved very slightly toward $x_0$ (which the model would
      have inferred to be $(1.0, -0.5)$). Multiplied over 500 more steps, the
      sample lands in the data distribution.</p>

      <h3>What $\\hat x_0$ means</h3>
      <p>At any timestep during sampling, we can compute the model's "running
      estimate of the clean data":</p>

      <div class="formula-box">$$\\hat x_0(x_t, t) \\;=\\; \\frac{1}{\\sqrt{\\bar\\alpha_t}}\\!\\left(x_t \\;-\\; \\sqrt{1 - \\bar\\alpha_t} \\, \\epsilon_\\theta(x_t, t)\\right)$$</div>

      <p>This is what the model "thinks" $x_0$ is, given the current noisy sample
      and predicted noise.</p>

      <p>$\\hat x_0(x_T, T)$ — at the start of sampling, with $x_T$ pure noise —
      is a noisy guess. $\\hat x_0(x_1, 1)$ — at the end — is a sharp estimate.
      Watching the <strong>evolution of $\\hat x_0$</strong> through sampling is
      one of the most pedagogically illuminating things you can do; it's how the
      paper's Figure 6 (CIFAR10 progressive generation) is computed.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'score-matching',
      toAnchor: 'langevin',
      toAnchorLabel: '6 — Langevin dynamics',
      body: `The $\\sigma_t z$ term is the same noise-injection step as in Langevin
             dynamics, with $\\sigma_t$ playing the role of $\\sqrt{2 \\eta}$.`,
    })}

    <div id="viz-sampling-chain" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
