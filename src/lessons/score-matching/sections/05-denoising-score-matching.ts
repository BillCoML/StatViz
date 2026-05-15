import { renderMath, crosslinkBack, proofToggle } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-5';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'dsm');
  sec.innerHTML = `
    <div class="section-label">§5</div>
    <h2>Denoising Score Matching (Vincent)</h2>
    <div class="prose">
      <p>Vincent (2011) found a different way around the
      $\\nabla \\log p_{\\mathrm{data}}$ problem: <strong>don't match the data score;
      match the score of a noise-corrupted version.</strong></p>

      <h3>Setup</h3>
      <p>Add Gaussian noise to data points:</p>
      $$\\tilde{x} = x + \\sigma\\varepsilon, \\qquad x \\sim p_{\\mathrm{data}},
      \\quad \\varepsilon \\sim \\mathcal{N}(0, I)$$

      <p>The noisy variable $\\tilde{x}$ follows the <strong>noise-perturbed
      distribution</strong> $p_\\sigma$:</p>
      $$p_\\sigma(\\tilde{x}) = \\int p_{\\mathrm{data}}(x)\\, \\mathcal{N}(\\tilde{x};\\, x, \\sigma^2 I)\\, dx$$

      <p>$p_\\sigma$ is $p_{\\mathrm{data}}$ convolved with a Gaussian kernel of width $\\sigma$.
      As $\\sigma \\to 0$ it converges to $p_{\\mathrm{data}}$; as $\\sigma \\to \\infty$ it
      becomes a wide Gaussian centered at the data mean.</p>

      <p>Our model $s_\\theta(\\tilde{x}, \\sigma)$ now takes both the noisy input
      <strong>and</strong> the noise level $\\sigma$, and tries to match the score of
      $p_\\sigma$. But Vincent's beautiful identity rescues us from needing
      $\\nabla \\log p_\\sigma$.</p>

      <h3>Vincent's identity</h3>
      <div class="formula-box">
        $$\\mathbb{E}_{\\tilde{x} \\sim p_\\sigma}\\!\\left[\\|s_\\theta(\\tilde{x}, \\sigma)
        - \\nabla_{\\tilde{x}} \\log p_\\sigma(\\tilde{x})\\|^2\\right]
        = \\mathbb{E}_{x,\\,\\varepsilon}\\!\\left[\\|s_\\theta(\\tilde{x}, \\sigma)
        - \\nabla_{\\tilde{x}} \\log q_\\sigma(\\tilde{x} \\mid x)\\|^2\\right] + C$$
      </div>

      <p>where $C$ doesn't depend on $\\theta$ and
      $q_\\sigma(\\tilde{x} \\mid x) = \\mathcal{N}(\\tilde{x};\\, x, \\sigma^2 I)$.
      The conditional score is just the score of a Gaussian:</p>

      $$\\nabla_{\\tilde{x}} \\log \\mathcal{N}(\\tilde{x};\\, x, \\sigma^2 I)
      = -\\frac{\\tilde{x} - x}{\\sigma^2} = -\\frac{\\varepsilon}{\\sigma}$$

      <p>Substituting:</p>
      <div class="formula-box">
        $$\\mathcal{L}_{\\mathrm{DSM}}(\\theta;\\, \\sigma)
        = \\mathbb{E}_{x,\\,\\varepsilon}\\!\\left[\\,\\left\\|s_\\theta(x + \\sigma\\varepsilon, \\sigma)
        + \\frac{\\varepsilon}{\\sigma}\\right\\|^2\\right]$$
      </div>

      <p><strong>The model takes a noisy input $\\tilde{x} = x + \\sigma\\varepsilon$
      and learns to predict $-\\varepsilon / \\sigma$.</strong> No Jacobian. No trace.
      Just MSE between two vectors.</p>

      ${proofToggle('Why this works — proof sketch', `
        <p>Expand the left side, focusing on the cross term:</p>
        $$-2 \\, \\mathbb{E}_{p_\\sigma}[s_\\theta(\\tilde{x}, \\sigma)^\\top \\nabla \\log p_\\sigma(\\tilde{x})]$$
        <p>Write $p_\\sigma(\\tilde{x}) = \\int p_{\\mathrm{data}}(x) q_\\sigma(\\tilde{x} \\mid x)\\,dx$
        and apply $p_\\sigma \\nabla \\log p_\\sigma = \\nabla p_\\sigma$. Substitute the integral
        expression for $p_\\sigma$ and move the gradient inside:</p>
        $$= \\int \\int p_{\\mathrm{data}}(x) q_\\sigma(\\tilde{x} \\mid x)
        s_\\theta(\\tilde{x}, \\sigma)^\\top \\nabla_{\\tilde{x}} \\log q_\\sigma(\\tilde{x} \\mid x)
        \\, dx\\, d\\tilde{x}$$
        $$= \\mathbb{E}_{x \\sim p_{\\mathrm{data}},\\, \\tilde{x} \\sim q_\\sigma(\\cdot \\mid x)}\\!
        \\left[s_\\theta(\\tilde{x}, \\sigma)^\\top \\nabla \\log q_\\sigma(\\tilde{x} \\mid x)\\right]$$
        <p>So the cross term in $\\mathcal{L}_\\sigma$ — which originally had $\\nabla \\log p_\\sigma$
        — equals the cross term with $\\nabla \\log q_\\sigma(\\tilde{x} \\mid x)$ instead. The
        other terms match (with $C$ absorbing the $\\|\\nabla \\log p_\\sigma\\|^2 -
        \\|\\nabla \\log q_\\sigma\\|^2$ discrepancy). $\\blacksquare$</p>
      `, false)}

      <h3>What the model learns to predict</h3>
      <p>Three equivalent parameterizations:</p>
      <ul>
        <li><strong>Score parameterization:</strong> $s_\\theta(\\tilde{x}, \\sigma)$
        directly outputs $\\approx \\nabla \\log p_\\sigma(\\tilde{x})$. Target: $-\\varepsilon / \\sigma$.</li>
        <li><strong>Noise parameterization</strong> ($\\varepsilon$-prediction): network
        $\\varepsilon_\\theta(\\tilde{x}, \\sigma)$ outputs $\\approx \\varepsilon$. Score recovered
        as $s_\\theta = -\\varepsilon_\\theta / \\sigma$.
        <strong>DDPM uses this parameterization.</strong></li>
        <li><strong>Clean-data parameterization:</strong> network $x_\\theta(\\tilde{x}, \\sigma)$
        outputs $\\approx x$. Score recovered as
        $s_\\theta = (x_\\theta - \\tilde{x}) / \\sigma^2$.</li>
      </ul>
      <p>All three are mathematically equivalent up to a per-$\\sigma$ rescaling.
      For DDPM, $\\varepsilon$-prediction with per-timestep weighting of 1 was empirically best.</p>

      <h3>Worked numerical example</h3>
      <p>Clean point $x = (0.5, 1.0)$, noise level $\\sigma = 0.1$,
      noise $\\varepsilon = (0.3, -0.5)$:</p>
      <ul>
        <li>Noisy input: $\\tilde{x} = (0.53,\\, 0.95)$.</li>
        <li>Score target: $-\\varepsilon / \\sigma = (-3,\\, 5)$.</li>
        <li>Loss at a zero-initialized model: $\\|(0,0) - (-3,5)\\|^2 = 9 + 25 = 34$.</li>
      </ul>
    </div>

    ${crosslinkBack({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'reparam-matrix',
      toAnchorLabel: '4 — reparameterization',
      body: `The construction $\\tilde{x} = x + \\sigma\\varepsilon$ is the
             reparameterization trick: a deterministic transform of fixed noise
             $\\varepsilon$, with $x$ acting as the mean and $\\sigma$ as the scale.
             This is what makes the DSM loss differentiable in any downstream
             pipeline that needs $\\partial \\mathcal{L} / \\partial x$.`,
    })}

    <div id="viz-noise-smoothed-score" class="viz-placeholder">
      <p class="viz-placeholder__label">Visualization: Noise-Smoothed Score (σ slider)</p>
    </div>
    <div id="viz-dsm-target" class="viz-placeholder">
      <p class="viz-placeholder__label">Visualization: DSM Target (click to place noisy point)</p>
    </div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
