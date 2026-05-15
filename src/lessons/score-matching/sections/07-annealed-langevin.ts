import { renderMath, crosslinkForward } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-7';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'annealed-langevin');
  sec.innerHTML = `
    <div class="section-label">§7</div>
    <h2>Annealed Langevin and the Path to DDPM</h2>
    <div class="prose">
      <p>The fix to Langevin's failure modes is <strong>anneal the noise level</strong>:
      train and sample at a sequence of decreasing $\\sigma$ values. Song & Ermon (2019)
      introduced this as <strong>Noise Conditional Score Networks (NCSN)</strong>.</p>

      <h3>Training: score at many noise levels</h3>
      <p>Train $s_\\theta(x, \\sigma)$ such that for every $\\sigma$ in a chosen schedule
      $\\{\\sigma_1 > \\sigma_2 > \\cdots > \\sigma_L\\}$:</p>
      $$s_\\theta(x, \\sigma_\\ell) \\approx \\nabla_x \\log p_{\\sigma_\\ell}(x)$$

      <p>Training loss — a weighted sum of DSM losses, one per noise level:</p>
      <div class="formula-box">
        $$\\mathcal{L}(\\theta) = \\sum_{\\ell=1}^{L} \\lambda(\\sigma_\\ell) \\cdot
        \\mathbb{E}_{x,\\,\\varepsilon}\\!\\left[\\,\\left\\|s_\\theta(x + \\sigma_\\ell\\varepsilon, \\sigma_\\ell)
        + \\frac{\\varepsilon}{\\sigma_\\ell}\\right\\|^2\\right]$$
      </div>
      <p>The weighting $\\lambda(\\sigma) = \\sigma^2$ rescales each term so all noise
      levels contribute at comparable scale. One <strong>network</strong> outputs scores
      at all noise levels by conditioning on $\\log\\sigma$.</p>

      <h3>Sampling: annealed Langevin</h3>
      <pre class="sm-pseudocode"><code>x ~ N(0, sigma_1^2 * I)

for ell = 1 to L:
    alpha_ell = epsilon * (sigma_ell / sigma_L)^2

    for t = 1 to T:
        eps_t ~ N(0, I)
        x = x + (alpha_ell / 2) * s_theta(x, sigma_ell)
              + sqrt(alpha_ell) * eps_t

return x   # approximately p_data when sigma_L is small</code></pre>

      <p>The intuition for each phase:</p>
      <ul>
        <li><strong>Early (large $\\sigma$):</strong> smooth score field defined everywhere;
        rapidly transports particles from initialization toward the rough vicinity of the
        data.</li>
        <li><strong>Middle:</strong> noise level shrinks; score develops structure; particles
        localize toward modes.</li>
        <li><strong>Late (small $\\sigma$):</strong> sharp score field; particles refine to
        high precision near the data manifold.</li>
      </ul>

      <h3>NCSN, DDPM, and the modern landscape</h3>
      <ul>
        <li><strong>NCSN (Song & Ermon 2019):</strong> train one $s_\\theta(x, \\sigma)$
        at $L$ discrete noise levels; sample by annealed Langevin.</li>
        <li><strong>DDPM (Ho et al. 2020):</strong> train one $\\varepsilon_\\theta(x_t, t)$
        at $T$ discrete timesteps. The forward process is a Markov chain with specific
        structure; the sampling process is the reverse Markov chain — Langevin-like at
        each step. The training loss simplifies to a re-weighted denoising score matching
        loss. <strong>Next lesson.</strong></li>
        <li><strong>Score-based SDEs (Song et al. 2021):</strong> take the continuous-time
        limit; the noise schedule becomes a continuous SDE. Unifies NCSN and DDPM.</li>
      </ul>
      <p>The differences are in parameterization, loss weighting, and discretization —
      not in the underlying idea. <strong>The idea is what §5 established: model the score
      at many noise levels, sample by walking down the noise schedule.</strong></p>
    </div>

    <div id="viz-annealed-langevin" class="viz-placeholder">
      <p class="viz-placeholder__label">Visualization: Annealed Langevin (centerpiece)</p>
    </div>

    ${crosslinkForward({
      toLesson: 'ddpm',
      body: `The DDPM training loss is exactly this — denoising score matching at multiple
             noise levels — with a specific parameterization (predict $\\varepsilon$, not the
             score) and a specific noise schedule (the $\\bar{\\alpha}_t$ schedule).
             The DDPM sampling process is annealed Langevin, derived from a Markov-chain
             perspective.`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
}
