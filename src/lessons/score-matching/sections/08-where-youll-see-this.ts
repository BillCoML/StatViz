import { renderMath, crosslinkBack, crosslinkForward, mountRoadmapMini } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-8';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§8</div>
    <h2>Where You'll See This</h2>
    <div class="prose">
      <p>Score-based modeling is the conceptual scaffolding for nearly all modern
      generative models.</p>

      <h3>Coming next</h3>
      <ul>
        <li><strong>DDPM</strong> — the canonical diffusion paper. The training loss
        reduces to denoising score matching at $T$ timesteps. The sampling process is
        reverse-time annealed Langevin.</li>
      </ul>

      <h3>Adjacent / sidebar</h3>
      <ul>
        <li><strong>Score-based SDEs</strong> (Song et al. 2021) — continuous-time view.
        Unifies NCSN and DDPM as discretizations of a single SDE.</li>
        <li><strong>EDM</strong> (Karras et al. 2022) — refined noise schedules,
        $\\sigma$-conditioning, and a unified parameterization.</li>
        <li><strong>Flow matching</strong> (Lipman et al. 2023) — bridges score matching
        and normalizing flows. Trains a vector field, not a score.</li>
        <li><strong>Energy-based models</strong> — the framework where score matching
        originated. Models specify $\\tilde{p}(x) = e^{-U(x)}$ directly.</li>
        <li><strong>Langevin sampling in molecular dynamics</strong> — the same SDE
        used in physical simulations of molecular motion.</li>
      </ul>

      <h3>Incoming references</h3>
      <ul>
        <li><strong>KL &amp; Jensen</strong> — Fisher divergence (§3) is connected to KL
        via De Bruijn's identity.</li>
        <li><strong>Gaussian Cookbook</strong> — the score of a Gaussian (§2), the
        reparameterization construction (§5), and the noise-smoothed density all use
        Cookbook identities.</li>
      </ul>
    </div>

    ${crosslinkBack({
      toLesson: 'kl-jensen',
      toAnchor: 'gibbs-inequality',
      toAnchorLabel: 'Non-negativity of KL',
      body: `Fisher divergence (§3) is non-negative for the same fundamental reason as
             KL — both measure the mismatch between two distributions. De Bruijn's
             identity makes the connection exact: $\\mathcal{F}$ is the derivative of
             $D_{\\mathrm{KL}}$ through noise smoothing.`,
    })}

    ${crosslinkBack({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'reparam-matrix',
      toAnchorLabel: '4 — reparameterization',
      body: `The denoising score matching construction $\\tilde{x} = x + \\sigma\\varepsilon$
             is the reparameterization trick from the Cookbook, applied here to define
             the noise corruption process.`,
    })}

    ${crosslinkForward({
      toLesson: 'ddpm',
      body: `DDPM takes the denoising score matching loss at $T$ noise levels and packages
             it into a forward/reverse Markov chain. After this lesson, you have everything
             you need to read the DDPM paper.`,
    })}

    <div id="roadmap-mini"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountRoadmapMini(sec.querySelector('#roadmap-mini') as HTMLElement, {
    currentLessonId: 'score-matching',
  });
}
