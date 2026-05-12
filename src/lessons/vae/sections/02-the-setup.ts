import { renderMath, crosslinkBack } from '@shared/ui';
import { mountEncoderDecoderDiagram } from '../viz/encoder-decoder-diagram';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-2';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§2</div>
    <h2>The Setup</h2>
    <div class="prose">
      <h3>The generative model</h3>
      <p>A VAE specifies a generative process in two steps:</p>
      <ol>
        <li><strong>Sample a latent</strong>: $z \\sim p(z) = \\mathcal{N}(0, I_d)$
        for some small latent dimension $d$.</li>
        <li><strong>Decode to data</strong>: $x \\sim p_\\theta(x \\mid z)$, where
        $p_\\theta$ is parameterized by a <strong>decoder neural network</strong>
        with parameters $\\theta$.</li>
      </ol>

      <p>For continuous data, the decoder output is Gaussian:
      $p_\\theta(x \\mid z) = \\mathcal{N}(\\mu_\\theta(z), \\sigma_x^2 I)$
      — the network predicts the mean of a Gaussian; the variance $\\sigma_x^2$ is a
      fixed hyperparameter. The model is <em>completely specified</em> by the prior
      $p(z)$ and the decoder $p_\\theta(x \\mid z)$. The encoder, introduced next,
      is a tool for <em>fitting</em> the model, not part of the generative story.</p>

      <h3>The learning objective</h3>
      <p>Given data $\\{x_1, \\ldots, x_N\\}$, maximize log-likelihood:</p>
      $$\\theta^* \\;=\\; \\arg\\max_\\theta \\sum_{i=1}^{N} \\log p_\\theta(x_i)
        \\;=\\; \\arg\\max_\\theta \\sum_{i=1}^{N} \\log \\int p_\\theta(x_i \\mid z) \\, p(z) \\, dz$$
      <p>Each integral is intractable: $z$ is continuous, $p_\\theta(x \\mid z)$ is a
      neural network, no closed form exists. This is exactly the latent-variable
      intractability encountered in ELBO &amp; VI.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'elbo-vi',
      toAnchor: 'fundamental-identity',
      toAnchorLabel: '3',
      body: `<p>The fundamental identity log p(x) = ELBO + KL developed in ELBO &amp; VI is
        the bridge between the intractable log-likelihood above and the objective we will
        actually optimize. The derivation carries over unchanged; only the decoder
        $p_\\theta$ is now a neural network rather than a fixed model.</p>`,
    })}

    <div id="viz-encoder-decoder-diagram" class="viz-medium"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountEncoderDecoderDiagram(sec.querySelector('#viz-encoder-decoder-diagram') as HTMLElement);
}
