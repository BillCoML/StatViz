import { renderMath, mountRoadmapMini } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-8';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§8</div>
    <h2>Where You'll See This</h2>
    <div class="prose">
      <p>The ELBO and its identity are everywhere in modern probabilistic ML. A short
      list of where, with links into the lessons that build on this one.</p>

      <h3>Already used</h3>
      <ul>
        <li><strong>EM</strong>
          (<a href="/lessons/em/#monotonicity">revisited above</a>). EM is exact
          variational EM where the E-step achieves the bound's tightness.</li>
        <li><strong>KL &amp; Jensen</strong> (the foundations). The fundamental identity
          is Jensen-on-log; the gap is reverse KL; both come from there.</li>
      </ul>

      <h3>Coming next</h3>
      <ul>
        <li><strong>Variational Autoencoders.</strong> Form 2 of the ELBO is the VAE
        loss. The encoder amortizes $q_\\phi(z \\mid x)$ across data points. The
        reparameterization trick (§5c) makes the reconstruction differentiable. The KL
        term keeps latents structured.</li>
        <li><strong>DDPM.</strong> The DDPM training objective is a sum of per-timestep
        ELBOs. Each timestep contributes a KL between two Gaussians (using the closed form
        from KL &amp; Jensen). The chain rule of KL decomposes the joint ELBO across
        diffusion steps.</li>
      </ul>

      <h3>Adjacent ideas (sidebar)</h3>
      <ul>
        <li><strong>Mean-field VI</strong> for graphical models (CAVI, exact updates).
          Bishop ch. 10 has the canonical treatment.</li>
        <li><strong>Black-box VI</strong> for non-conjugate models
          (Ranganath, Gerrish, Blei 2014).</li>
        <li><strong>β-VAE</strong> and information-bottleneck variants — controlling the
          KL coefficient changes representation properties.</li>
        <li><strong>Importance-weighted ELBO (IWAE)</strong> — tighter bounds via
          multiple $q$-samples.</li>
      </ul>
    </div>

    <div id="roadmap-mini"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountRoadmapMini(sec.querySelector('#roadmap-mini') as HTMLElement, {
    currentLessonId: 'elbo-vi',
  });
}
