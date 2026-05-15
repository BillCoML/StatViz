import { renderMath } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-11';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§11</div>
    <h2>Practical Considerations</h2>
    <div class="prose">
      <p>Several design choices in the paper deserve discussion: the noise
      schedule, the variance parameterization $\\Sigma_\\theta$, the loss
      weighting (full $L$ vs $L_{\\text{simple}}$), and the practical
      trade-offs between sample quality and log likelihood.</p>

      <h3>Noise schedule choice</h3>
      <p>Paper uses <strong>linear $\\beta_t$ from $10^{-4}$ to $0.02$ over
      $T = 1000$</strong>. Considerations:</p>
      <ul>
        <li><strong>Endpoint $L_T \\approx 10^{-5}$ bits/dim</strong>. The signal
        is destroyed; $q(x_T \\mid x_0)$ is essentially $\\mathcal{N}(0, I)$.</li>
        <li><strong>Small $\\beta_t$ keeps the local Gaussian approximation
        valid</strong>. Reverse transitions are well-modeled as Gaussian only when
        the forward step is small.</li>
        <li><strong>Linear is empirically fine but not optimal</strong>. Cosine
        schedules (Nichol &amp; Dhariwal 2021) and others improve quality on
        harder datasets.</li>
      </ul>

      <h3>$\\Sigma_\\theta$: learned vs fixed</h3>
      <p>The paper's ablation (Table 2): fixing $\\Sigma_\\theta = \\sigma_t^2 I$
      with $\\sigma_t^2 \\in \\{\\beta_t, \\tilde\\beta_t\\}$ works <strong>better</strong>
      than learning a diagonal $\\Sigma_\\theta(x_t, t)$. Learning the variance
      leads to instability.</p>

      <p>Why? Empirically: the variance learning interacts badly with the
      loss-weight-flattening of $L_{\\text{simple}}$. Better-quality theoretical
      treatments (e.g., Nichol &amp; Dhariwal's IDDPM, 2021) learn an
      <em>interpolation coefficient</em> between $\\beta_t$ and $\\tilde\\beta_t$
      — a single scalar per $t$ — and recover the stability while gaining
      flexibility.</p>

      <h3>$L$ vs $L_{\\text{simple}}$: the trade-off</h3>
      <p>Paper's Table 1: training on the full $L$ gives better
      <strong>codelengths</strong> (negative log likelihood); training on
      $L_{\\text{simple}}$ gives better <strong>sample quality</strong> (FID).</p>

      <p>Why the divergence?</p>
      <ul>
        <li><strong>Full $L$</strong> matches the variational objective exactly.
        The model is optimally trained as a density estimator.</li>
        <li><strong>$L_{\\text{simple}}$</strong> drops the per-timestep weights,
        which re-balances the training signal. The model spends more capacity on
        harder (large-$t$) denoising tasks. This is suboptimal for density
        estimation but better for the perceptual quality of samples.</li>
      </ul>

      <p>The choice depends on the application. If you want a compressor that
      achieves good bits/dim: full $L$. If you want pretty pictures:
      $L_{\\text{simple}}$.</p>

      <h3>Progressive lossy compression</h3>
      <p>The paper's §4.3 reframes the trained model as a <strong>progressive
      lossy compressor</strong>. Algorithms 3 and 4 in the paper transmit a sample
      using $\\approx D_{\\mathrm{KL}}(q(x) \\| p(x))$ bits, with the receiver
      progressively decoding from coarse (large $t$) to fine (small $t$). The
      rate-distortion curve (Figure 5 of the paper): most of the bits are spent
      on imperceptible details.</p>

      <p>This isn't a deployable compressor (it requires minimal random coding,
      which isn't tractable), but it's a useful conceptual framing.
      <strong>Diffusion models are progressive decoders</strong>, and this
      structure is what makes them so good at generating coherent global
      structure first and fine details last.</p>

      <h3>Variants and extensions (sketched, links out)</h3>
      <ul>
        <li><strong>DDIM</strong> (Song et al. 2021): same trained network,
        deterministic sampler ($\\sigma_t = 0$). 10× faster sampling at slightly
        lower quality.</li>
        <li><strong>Classifier guidance</strong> (Dhariwal &amp; Nichol 2021):
        condition the sampling on a class label.</li>
        <li><strong>Classifier-free guidance</strong> (Ho &amp; Salimans 2021):
        train a conditional and unconditional model jointly; combine at sampling
        for tunable conditional strength. The basis of all modern conditional
        diffusion.</li>
        <li><strong>Latent diffusion</strong> (Rombach et al. 2022): run the
        diffusion in the latent space of a pre-trained autoencoder. The basis of
        Stable Diffusion.</li>
        <li><strong>Score-based SDEs</strong> (Song et al. 2021): the
        continuous-time limit.</li>
      </ul>
    </div>

    <div id="viz-noise-schedule-explorer" style="margin-top:1.5rem;"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
