import { renderMath } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-10';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§10</div>
    <h2>A Trained 2D DDPM You Can Explore</h2>
    <div class="prose">
      <p>The §9 sampling chain is one mode of interaction. This section turns the
      trained 2D DDPM into a <strong>full explorer</strong>: forward diffusion,
      reverse sampling, $\\hat x_0$ trajectories, interpolation in latent space
      (recreating Figures 8 and 9 of the paper), and direct comparison with VAE
      and Score Matching outputs.</p>

      <p>The model: 2D synthetic data (the same 4-cluster mixture used throughout
      the curriculum), $T = 100$ timesteps (small enough for 30 fps animation),
      $\\epsilon_\\theta$ implemented as an MLP that takes $(x_t, t)$ where $t$
      is sinusoidally embedded. ~5000 parameters total. Pre-trained offline;
      weights ship as JSON.</p>

      <h3>What the explorer lets you do</h3>
      <ul>
        <li><strong>Mode 1 — Forward chain.</strong> Pick a data point. Watch it
        diffuse over $T$ steps into noise.</li>
        <li><strong>Mode 2 — Reverse sampling.</strong> Pick a noise sample.
        Animate the reverse process. Watch the sample evolve from pure noise to a
        data-distribution sample.</li>
        <li><strong>Mode 3 — $\\hat x_0$ evolution.</strong> Run a reverse-sampling
        chain and visualize the model's $\\hat x_0$ estimate at every step. The
        $\\hat x_0$ trajectory traces a path from "a generic guess at the data
        center" to a specific cluster.</li>
        <li><strong>Mode 4 — Latent interpolation (Figure 8 recreation).</strong>
        Pick two data points. The explorer diffuses both to timestep $t^*$,
        linearly interpolates, runs the reverse process, and plots the outputs.
        At small $t^*$: interpolations stay near the source. At large $t^*$:
        interpolations are essentially independent samples.</li>
        <li><strong>Mode 5 — Score field overlay.</strong> At any $t$, overlay the
        learned score field $s_\\theta = -\\epsilon_\\theta / \\sqrt{1 - \\bar\\alpha_t}$.
        Directly visualizes the §7 equivalence.</li>
        <li><strong>Mode 6 — Cross-model comparison.</strong> Same data, three
        trained models (DDPM, VAE, Score Matching). Side-by-side panels.</li>
      </ul>
    </div>

    <div id="viz-trained-ddpm-explorer" style="margin-top:1.5rem;"></div>
    <div id="viz-coarse-to-fine-interp" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
