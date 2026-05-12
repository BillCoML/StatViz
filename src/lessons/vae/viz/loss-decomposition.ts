export function mountLossDecomposition(container: HTMLElement): void {
  container.innerHTML = `
    <div class="vae-loss-demo">
      <div style="font-family:var(--font-display);font-weight:600;margin-bottom:1rem;">
        Trade-off: reconstruction vs KL
      </div>

      <div style="display:flex;flex-direction:column;gap:0.5rem;margin-bottom:1rem;">
        <label class="viz-label">
          Reconstruction quality
          <input type="range" id="loss-recon-slider" min="0" max="100" value="50" step="1">
          <span id="loss-recon-val" style="font-family:var(--font-mono);font-size:0.85em;min-width:3em;">50</span>
        </label>
        <label class="viz-label">
          KL closeness to prior
          <input type="range" id="loss-kl-slider" min="0" max="100" value="50" step="1">
          <span id="loss-kl-val" style="font-family:var(--font-mono);font-size:0.85em;min-width:3em;">50</span>
        </label>
      </div>

      <div class="vae-loss-bar-row">
        <div style="font-size:0.78em;color:var(--ink-soft);margin-bottom:0.25rem;">Total −ELBO (loss)</div>
        <div class="vae-loss-bar" id="loss-bar">
          <div class="vae-loss-bar__recon" id="loss-bar-recon" style="width:50%;"></div>
          <div class="vae-loss-bar__kl"    id="loss-bar-kl"    style="width:10%;"></div>
        </div>
        <div class="vae-loss-legend">
          <span><span class="vae-loss-legend__swatch" style="background:var(--recon);"></span>Reconstruction loss</span>
          <span><span class="vae-loss-legend__swatch" style="background:var(--kl-reg);"></span>KL loss</span>
        </div>
      </div>

      <div id="loss-annotation" style="font-family:var(--font-serif);font-style:italic;font-size:0.9em;
           color:var(--ink-soft);margin-top:0.75rem;min-height:1.4em;transition:color 0.2s;">
        Balanced trade-off — a healthy VAE.
      </div>
    </div>
  `;

  const reconSlider = container.querySelector('#loss-recon-slider') as HTMLInputElement;
  const klSlider    = container.querySelector('#loss-kl-slider') as HTMLInputElement;
  const reconVal    = container.querySelector('#loss-recon-val') as HTMLElement;
  const klVal       = container.querySelector('#loss-kl-val') as HTMLElement;
  const reconBar    = container.querySelector('#loss-bar-recon') as HTMLElement;
  const klBar       = container.querySelector('#loss-bar-kl') as HTMLElement;
  const annotation  = container.querySelector('#loss-annotation') as HTMLElement;

  function update() {
    const r = +reconSlider.value;
    const k = +klSlider.value;
    reconVal.textContent = String(r);
    klVal.textContent = String(k);

    // recon loss is inverse of quality; kl loss is inverse of closeness
    const reconLoss = 100 - r;
    const klLoss    = 100 - k;
    const total     = reconLoss + klLoss;
    const rFrac     = total > 0 ? (reconLoss / total) * 80 : 40;
    const kFrac     = total > 0 ? (klLoss / total) * 80 : 5;

    reconBar.style.width = rFrac + '%';
    klBar.style.width    = kFrac + '%';

    if (r < 30 && k > 70) {
      annotation.textContent = 'Posterior collapse — KL ≈ 0, encoder ignores x. Blurry samples.';
      annotation.style.color = 'var(--kl-reg)';
    } else if (r > 70 && k < 30) {
      annotation.textContent = 'Reconstruction collapse — encoder overfits, prior samples decode to garbage.';
      annotation.style.color = 'var(--kl-reg)';
    } else if (r > 70 && k > 70) {
      annotation.textContent = 'Healthy VAE — good reconstruction and structured latent space.';
      annotation.style.color = 'var(--encoder)';
    } else {
      annotation.textContent = 'Balanced trade-off — a healthy VAE.';
      annotation.style.color = 'var(--ink-soft)';
    }
  }

  reconSlider.addEventListener('input', update);
  klSlider.addEventListener('input', update);
  update();
}
