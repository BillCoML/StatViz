import { renderMath, mountRoadmapMini, crosslinkForward } from '@shared/ui';

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

    ${crosslinkForward({
      toLesson: 'gaussian-cookbook',
      toAnchor: 'kl-mvn-diag',
      toAnchorLabel: '3',
      body: `<p>The KL regularizer in the VAE ELBO — $D_{\\mathrm{KL}}(q_\\phi(z \\mid x) \\| p(z))$ with
        $q_\\phi$ diagonal Gaussian and $p(z) = \\mathcal{N}(0, I)$ — has a closed form derived
        in the Gaussian Cookbook. It's the identity every VAE implementation uses as a single
        line of the loss.</p>`,
    })}

    ${crosslinkForward({
      toLesson: 'vae',
      toAnchor: 'vae-objective',
      body: `<p>Form 2 of the ELBO — reconstruction minus KL regularizer — is the VAE training
        loss verbatim. The encoder amortizes $q_\\phi(z \\mid x)$ across data points; the
        reparameterization trick makes the reconstruction term differentiable.</p>`,
    })}

    ${crosslinkForward({
      toLesson: 'ddpm',
      toAnchor: 'vlb',
      body: `<p>DDPM extends Form 1 of the ELBO to a $T$-step hierarchical setting with a
        frozen Gaussian encoder. The decomposition into $L_T + \\sum L_{t-1} + L_0$ — a
        sum of closed-form Gaussian KLs — is exactly the variance-reduced ELBO of this
        lesson, run on a Markov chain.</p>`,
    })}

    <div id="roadmap-mini"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountRoadmapMini(sec.querySelector('#roadmap-mini') as HTMLElement, {
    currentLessonId: 'elbo-vi',
  });
}
