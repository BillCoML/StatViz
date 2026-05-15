import { renderMath, crosslinkSidebar } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-8';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'training-algorithm');
  sec.innerHTML = `
    <div class="section-label">§8</div>
    <h2 id="training-algorithm">The Training Algorithm</h2>
    <div class="prose">
      <p>The paper's Algorithm 1 is six lines:</p>

      <pre class="algorithm-block">Algorithm 1 — Training
1: repeat
2:    x_0 ~ q(x_0)                  // sample a clean data point
3:    t ~ Uniform({1, ..., T})       // pick a random timestep
4:    epsilon ~ N(0, I)              // sample noise
5:    Take gradient descent step on
         grad_theta || epsilon - epsilon_theta(sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon, t) ||^2
6: until converged</pre>

      <p>Six lines that took §1–§7 to motivate. Let me unpack them.</p>

      <h3>Line 2: $x_0 \\sim q(x_0)$</h3>
      <p>Sample a data point from the training distribution. In practice: a
      minibatch of images from the dataset.</p>

      <h3>Line 3: $t \\sim \\mathrm{Uniform}(\\{1, \\ldots, T\\})$</h3>
      <p><strong>Random timestep per data point.</strong> Why uniform? Because
      $L_{\\text{simple}}$ is the average of $L_{0}, L_{1}, \\ldots, L_{T-1}$ over
      $t$. Sampling $t$ uniformly gives an unbiased Monte Carlo estimate of the
      average. Each gradient step thus optimizes the <em>expected</em>
      per-timestep loss, drawn from a uniform distribution over timesteps.</p>

      <p>Alternative: importance-sample $t$ proportional to the per-timestep loss
      magnitude. The paper finds uniform sampling works well enough and is simpler.</p>

      <h3>Line 4: $\\epsilon \\sim \\mathcal{N}(0, I)$</h3>
      <p>Sample noise of the same shape as $x_0$. This is the noise that will
      define the forward sample $x_t$ and the prediction target.</p>

      <h3>Line 5: the gradient step</h3>
      <p>Compute
      $x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1 - \\bar\\alpha_t} \\epsilon$ via
      the <strong>closed-form jump</strong>. Pass through the network to get
      $\\epsilon_\\theta(x_t, t)$. Compute MSE against the true $\\epsilon$.
      Backprop. Step.</p>

      <p>Note: $x_0$ and $\\epsilon$ enter only through $x_t$ in the loss, but the
      <strong>gradient flows back through both</strong> via standard autograd. The
      model "sees" only the noisy sample; supervision comes from the exact
      $\\epsilon$ that produced it.</p>

      <h3>Tracing one step numerically</h3>
      <p>Take a 2D point $x_0 = (1.0, -0.5)$, timestep $t = 500$, noise
      $\\epsilon = (0.3, -0.7)$. For the paper's schedule:</p>
      <ul>
        <li>$\\sqrt{\\bar\\alpha_{500}} \\approx 0.279$</li>
        <li>$\\sqrt{1 - \\bar\\alpha_{500}} \\approx 0.960$</li>
      </ul>

      $$x_{500} \\;=\\; 0.279 \\cdot (1.0, -0.5) + 0.960 \\cdot (0.3, -0.7) \\;\\approx\\; (0.567, -0.812)$$

      <p>The model receives $(0.567, -0.812)$ and the integer $t = 500$, and is
      asked to output $(0.3, -0.7)$. Loss is $\\|\\epsilon - \\epsilon_\\theta\\|^2$.</p>

      <p>Over thousands of $(x_0, t, \\epsilon)$ triplets, the network learns to
      invert the forward process: given a noisy sample, what noise produced it?</p>
    </div>

    ${crosslinkSidebar({
      toLesson: 'elbo-vi',
      body: `Standard VI optimizes the full ELBO at each step. DDPM optimizes a random
             term — $L_{t-1}$ for random $t$ — at each step. This is a Monte Carlo
             estimate of the sum of $L_{t-1}$ over all $t$. The randomization is what
             makes the per-step gradient computation $O(1)$ instead of $O(T)$.`,
    })}

    <div id="viz-training-trace" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
