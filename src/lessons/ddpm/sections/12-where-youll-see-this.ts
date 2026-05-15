import { renderMath, mountRoadmapMini } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-12';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§12</div>
    <h2>Where You'll See This (and What's Next)</h2>
    <div class="prose">
      <p>You've reached the destination of StatViz.</p>

      <p>Diffusion models are, as of 2024, the dominant generative modeling
      paradigm for image, video, audio, and increasingly other modalities. The
      DDPM paper you've just read in full is the foundation. Modern systems
      extend it in several directions, but the core machinery is what's in this
      lesson.</p>

      <h3>Direct descendants</h3>
      <ul>
        <li><strong>Stable Diffusion</strong> (Rombach et al. 2022): DDPM in the
        latent space of a pre-trained VAE.</li>
        <li><strong>Imagen</strong> (Saharia et al. 2022) and <strong>DALL-E 2</strong>
        (Ramesh et al. 2022): text-to-image diffusion at scale. Classifier-free
        guidance on a text-conditioned $\\epsilon_\\theta(x_t, t, c)$.</li>
        <li><strong>Video diffusion</strong> (Ho et al. 2022): factorize space and
        time; diffusion over a 3D tensor. Sora and friends.</li>
        <li><strong>Audio diffusion</strong>: AudioLM, Riffusion, Suno. Direct
        application to 1D signals.</li>
      </ul>

      <h3>Theoretical/mathematical extensions</h3>
      <ul>
        <li><strong>Score-based SDEs</strong> (Song et al. 2021): continuous-time
        limit; the forward process is an SDE, the reverse is its time-reversed
        SDE.</li>
        <li><strong>Flow matching</strong> (Lipman et al. 2023): an ODE-based
        alternative to diffusion.</li>
        <li><strong>Schrödinger bridges</strong> (De Bortoli et al. 2021): the
        most general formulation, of which DDPM is a special case.</li>
        <li><strong>Consistency models</strong> (Song et al. 2023): few-step
        (sometimes single-step) sampling.</li>
      </ul>

      <h3>Where you can go from here</h3>
      <p>You're now equipped to read essentially any modern generative modeling
      paper. The natural follow-ons:</p>
      <ol>
        <li><strong>Score-based SDEs</strong> (Song et al. 2021).</li>
        <li><strong>Latent diffusion</strong> (Rombach et al. 2022).</li>
        <li><strong>Classifier-free guidance</strong> (Ho &amp; Salimans 2021).</li>
        <li><strong>Flow matching</strong> (Lipman et al. 2023).</li>
        <li>Sidebars mentioned earlier in the curriculum: Normalizing Flows, MCMC,
        Langevin dynamics (standalone). All now reachable.</li>
      </ol>
    </div>

    <div class="ddpm-final">
      <div class="ddpm-final__msg">
        You've finished StatViz. The DDPM paper is now readable to you in its
        entirety. Go read it again — it'll feel very different this time.
      </div>
    </div>

    <div id="viz-final-roadmap" style="margin-top:1.5rem;"></div>

    <div id="roadmap-mini" style="margin-top:2rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountRoadmapMini(sec.querySelector('#roadmap-mini') as HTMLElement, {
    currentLessonId: 'ddpm',
  });
}
