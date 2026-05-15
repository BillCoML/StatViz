import { renderMath, mountPrereqStrip } from '@shared/ui';
import { meta } from '../meta';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-1';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="ddpm-hook">
      <h1 class="ddpm-hook__title">Denoising Diffusion Probabilistic Models</h1>
      <p class="ddpm-hook__subtitle">A hierarchical VAE, a score-matching model, and an annealed Langevin sampler — all the same thing.</p>
      <div class="prose">
        <p>Diffusion models do something that, the first time you see it, looks like
        a magic trick. Start with pure Gaussian noise — a canvas of static. Apply a
        neural network 1000 times in sequence, each time nudging the image slightly.
        At the end, the static has become a photorealistic human face that has never
        existed.</p>

        <p>The training story is just as strange. The model is never shown a single
        complete image alongside a goal. It only ever sees a real image with some
        Gaussian noise added, and it's asked: <em>what noise was added?</em> That's
        the entire training signal.</p>

        <p>How does this work? <strong>Two answers, both true.</strong></p>

        <p><strong>First answer (variational):</strong> A diffusion model is a
        hierarchical VAE. It has $T = 1000$ latent variables $x_1, \\ldots, x_T$,
        arranged in a Markov chain. The "encoder" is frozen: it just adds Gaussian
        noise at each step until pure noise remains. The "decoder" — the reverse
        Markov chain $p_\\theta(x_{t-1} \\mid x_t)$ — is what gets learned, by
        gradient descent on the ELBO.</p>

        <p><strong>Second answer (score matching):</strong> A diffusion model learns
        $\\nabla_x \\log p(x_t)$ at every noise level $t$, then samples by walking
        down the noise schedule with Langevin dynamics. The training loss is
        denoising score matching. The sampling procedure is annealed Langevin.</p>

        <p><strong>Both answers describe the same object.</strong> The paper's central
        contribution (Ho, Jain, Abbeel 2020) is showing that the right
        parameterization of the variational model makes it literally equivalent to
        the score model. This lesson develops both views in parallel and lands on
        the equivalence as a punchline.</p>
      </div>
      <a class="ddpm-cta" href="#section-2">Watch the forward process →</a>
      <div id="prereq-strip-container" style="margin-top:2rem;"></div>
      <div id="viz-hero-animation" style="margin-top:2rem;"></div>
    </div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountPrereqStrip(
    sec.querySelector('#prereq-strip-container') as HTMLElement,
    { prerequisites: meta.prerequisites },
  );
}
