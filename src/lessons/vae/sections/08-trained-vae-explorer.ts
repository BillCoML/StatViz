import { renderMath } from '@shared/ui';
import { mountTrainedVAEExplorer } from '../viz/trained-vae-explorer';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-8';
  sec.className = 'section';
  sec.innerHTML = `
    <div id="latent-interpolation"></div>
    <div class="section-label">§8</div>
    <h2>A Trained VAE You Can Explore</h2>
    <div class="prose">
      <p>Now let us look at a real (tiny) VAE trained on a 2D four-cluster dataset.
      The architecture is the same as §7 but with hidden dim 16. We trained it for
      4000 epochs on 1000 samples from a mixture of four Gaussians
      at $\\pm(2, \\pm 2)$, at six values of $\\beta$. The weights live in
      <code>vae-weights.json</code>; the forward passes run here in your browser.</p>

      <h3>What to look for</h3>
      <ol>
        <li><strong>Encoding</strong>: hover a data point. Watch its encoded distribution
        appear as an ellipse in the latent plot.</li>
        <li><strong>Latent structure</strong> ($\\beta = 1$): each cluster in data space lands
        in a coherent region of latent space, all four regions cluster near the
        origin. This is the prior regularizer working.</li>
        <li><strong>Decoding</strong>: click anywhere in latent space. The decoder produces
        a point in data space. Near the origin you get plausible data; far from
        the origin the decoder extrapolates.</li>
        <li><strong>Sampling</strong>: click "Sample from prior" to draw
        $z \\sim \\mathcal{N}(0, I)$ and decode. Repeated clicks produce diverse
        plausible data.</li>
        <li><strong>Interpolation</strong>: click two latent points, then "Interpolate."
        The decoder traces a smooth path through data space — the model has learned a
        continuous representation.</li>
        <li><strong>$\\beta$ slider</strong>: drag from 0.25 to 10 and watch the latent
        space rearrange. Higher $\\beta$ → tighter clustering. Lower $\\beta$ →
        spread out. Use §9 to understand the failure modes at the extremes.</li>
      </ol>
    </div>

    <div id="viz-trained-vae-explorer" class="viz-wide viz-centerpiece"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountTrainedVAEExplorer(sec.querySelector('#viz-trained-vae-explorer') as HTMLElement);
}
