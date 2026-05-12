import { renderMath, crosslinkForward, mountRoadmapMini } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-10';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§10</div>
    <h2>Where You'll See This</h2>
    <div class="prose">
      <p>The VAE is the gateway architecture for modern probabilistic generative
      modeling. Direct extensions and applications:</p>

      <h3>Coming next</h3>
      <ul>
        <li><strong>DDPM.</strong> A hierarchical VAE with $T$ layers of latents
        $z_1, \\ldots, z_T$ structured as a Markov chain. The encoder is <em>fixed</em>
        (a known Gaussian noising process); only the decoder is learned. The training
        objective is still a sum of ELBO terms, each of which is a KL between Gaussians
        from the Gaussian Cookbook. The patterns from §4–§6 here carry over almost
        unchanged.</li>
        <li><strong>Score Matching.</strong> An entirely different generative approach —
        no encoder, no latents. Learns the gradient of the log-density. Combines with
        VAE-like architectures in modern diffusion models.</li>
      </ul>

      <h3>Adjacent / sidebar</h3>
      <ul>
        <li><strong>VQ-VAE</strong> — discrete latents instead of continuous; lookup
        instead of sampling.</li>
        <li><strong>VAE-GAN hybrids</strong> — adversarial loss in addition to ELBO for
        sharper reconstructions.</li>
        <li><strong>Hierarchical / ladder VAEs</strong> — multiple latent layers,
        each capturing a different scale.</li>
        <li><strong>β-VAE and friends</strong> — the disentanglement literature.</li>
      </ul>
    </div>

    ${crosslinkForward({
      toLesson: 'ddpm',
      body: `<p>DDPM is a hierarchical VAE with a fixed Gaussian encoder (the forward
        noising chain) and a learned decoder (the reverse chain). Every formula
        from §4–§6 appears in the DDPM derivation, scaled across $T$ diffusion timesteps.</p>`,
    })}

    ${crosslinkForward({
      toLesson: 'score-matching',
      body: `<p>Score matching learns the gradient of the log-density — a complementary
        approach that skips latents entirely. The connection runs through Tweedie's formula
        and the denoising score matching objective.</p>`,
    })}

    <div id="roadmap-mini"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountRoadmapMini(sec.querySelector('#roadmap-mini') as HTMLElement, {
    currentLessonId: 'vae',
  });
}
