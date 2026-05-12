export function mountEncoderDecoderDiagram(container: HTMLElement): void {
  container.innerHTML = `
    <div class="vae-arch-diagram">
      <div class="vae-arch-box vae-arch-box--data">
        <div class="vae-arch-box__symbol" style="color:var(--data-x);">x</div>
        <div class="vae-arch-box__label">data</div>
        <div class="vae-arch-tooltip">Input data — e.g., a 2D point or a 784-dim image</div>
      </div>

      <div class="vae-arch-arrow">→</div>

      <div class="vae-arch-box vae-arch-box--encoder">
        <div class="vae-arch-box__symbol" style="color:var(--encoder);">q<sub>φ</sub></div>
        <div class="vae-arch-box__label">encoder</div>
        <div class="vae-arch-tooltip">MLP with params φ. Outputs μ_φ(x) and log σ_φ(x).</div>
      </div>

      <div class="vae-arch-arrow">→</div>

      <div style="display:flex;flex-direction:column;align-items:center;gap:0.25rem;">
        <div class="vae-arch-box vae-arch-box--latent">
          <div class="vae-arch-box__symbol" style="color:var(--latent-z);">z</div>
          <div class="vae-arch-box__label">latent sample</div>
          <div class="vae-arch-tooltip">z = μ + σ⊙ε via reparameterization. Differentiable!</div>
        </div>
        <div style="display:flex;align-items:center;gap:0.3rem;font-size:0.75em;color:var(--kl-reg);">
          <span>↑ KL ↑</span>
        </div>
        <div class="vae-arch-box" style="border-color:var(--prior);min-width:5rem;">
          <div class="vae-arch-box__symbol" style="color:var(--prior);">p(z)</div>
          <div class="vae-arch-box__label">prior N(0,I)</div>
          <div class="vae-arch-tooltip">The standard normal prior. KL regularizer pulls q_φ toward p(z).</div>
        </div>
      </div>

      <div class="vae-arch-arrow">→</div>

      <div class="vae-arch-box vae-arch-box--decoder">
        <div class="vae-arch-box__symbol" style="color:var(--decoder);">p<sub>θ</sub></div>
        <div class="vae-arch-box__label">decoder</div>
        <div class="vae-arch-tooltip">MLP with params θ. Outputs μ_θ(z) — the reconstruction mean.</div>
      </div>

      <div class="vae-arch-arrow">→</div>

      <div class="vae-arch-box vae-arch-box--data">
        <div class="vae-arch-box__symbol" style="color:var(--data-x);">x̂</div>
        <div class="vae-arch-box__label">reconstruction</div>
        <div class="vae-arch-tooltip">Decoded output. We want ‖x − x̂‖ to be small.</div>
      </div>
    </div>
    <p style="text-align:center;font-size:0.78em;color:var(--ink-soft);font-style:italic;margin-top:0.25rem;">
      Hover each box for details. Gradients flow through the encoder via the reparameterization trick.
    </p>
  `;
}
