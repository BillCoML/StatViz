import { renderMath, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-3';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'reverse-process');
  sec.innerHTML = `
    <div class="section-label">§3</div>
    <h2 id="reverse-process">The Reverse Process: A Hierarchical VAE</h2>
    <div class="prose">
      <p>The <strong>reverse process</strong> is what we learn. Define it as a
      Markov chain starting at the standard Gaussian and working backward:</p>

      <div class="formula-box">$$p_\\theta(x_{0:T}) \\;=\\; p(x_T) \\prod_{t=1}^{T} p_\\theta(x_{t-1} \\mid x_t), \\qquad p(x_T) \\;=\\; \\mathcal{N}(x_T; 0, I)$$</div>

      <p>Each reverse transition is parameterized as a Gaussian:</p>

      $$p_\\theta(x_{t-1} \\mid x_t) \\;=\\; \\mathcal{N}\\!\\big(x_{t-1}; \\; \\mu_\\theta(x_t, t), \\; \\Sigma_\\theta(x_t, t)\\big)$$

      <p>The neural network takes the current noisy sample $x_t$ and the timestep
      $t$, and outputs the mean (and possibly variance) of the Gaussian
      distribution for the previous step. To sample data, start at
      $x_T \\sim \\mathcal{N}(0, I)$ and walk backward through the chain.</p>

      <h3>Why Gaussian reverse transitions work</h3>
      <p>A subtle and important fact: when the forward step variance $\\beta_t$ is
      <strong>small</strong>, the reverse transition $q(x_{t-1} \\mid x_t)$ is
      approximately Gaussian — even though it's not Gaussian in general. This is a
      classical result from Feller (1949), referenced in Sohl-Dickstein et al.
      (2015). The paper exploits this: small $\\beta_t$ means the forward and
      reverse processes have <strong>the same functional form</strong> (Gaussian),
      justifying the parameterization choice.</p>

      <p>If $\\beta_t$ were large, the reverse step would be highly multimodal
      (which mode of the data did $x_t$ come from?) and a Gaussian wouldn't fit.
      With $\\beta_t \\leq 0.02$, the local Gaussian approximation is tight enough
      to drive a working sampler.</p>

      <h3>Reading DDPM as a VAE</h3>
      <p>Compare to the VAE setup:</p>
      <ul>
        <li><strong>VAE:</strong> Encoder $q_\\phi(z \\mid x)$ — learned, neural
        network. Single latent $z$. Decoder $p_\\theta(x \\mid z)$ — learned.</li>
        <li><strong>DDPM:</strong> Encoder $q(x_{1:T} \\mid x_0)$ — <strong>fixed</strong>,
        no learnable parameters (the noise schedule is the only "structure"). $T$
        latents $x_1, x_2, \\ldots, x_T$, of the <strong>same dimensionality</strong>
        as the data (this is crucial — unlike VAE where $z$ is low-dimensional).
        Decoder
        $p_\\theta(x_0 \\mid x_T) = \\int p(x_T) \\prod p_\\theta(x_{t-1} \\mid x_t) dx_{1:T-1}$
        — learned, parameterized as a chain.</li>
      </ul>

      <p><strong>DDPM is a hierarchical VAE with a frozen encoder and $T$
      same-dim latents arranged in a Markov chain.</strong> Everything you know
      about VAE training — the ELBO, the reparameterization trick, the variational
      objective — generalizes directly.</p>

      <p>The frozen encoder is what makes DDPM both more constrained and more
      powerful than a standard VAE. More constrained: we can't learn what kind of
      latent code is most useful — it's fixed to be "noisy versions of the data."
      More powerful: the constraint is the <em>right</em> constraint for image
      data, where additive Gaussian noise is geometrically natural.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'vae',
      toAnchor: 'vae-objective',
      body: `Everything below — the ELBO derivation, the reparameterization trick,
             the gradient-of-log-prob objective — extends VAE to a $T$-step hierarchical
             setting with a frozen encoder. If §4–§6 feel like VAE déjà vu, that's because
             they are.`,
    })}

    <div id="viz-graphical-model" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
