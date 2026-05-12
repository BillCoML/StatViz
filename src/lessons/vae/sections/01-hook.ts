import { renderMath, mountPrereqStrip } from '@shared/ui';
import { meta } from '../meta';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-1';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="vae-hook">
      <div class="vae-hook__pipeline" aria-hidden="true">
        <div class="vae-pipeline-box vae-pipeline-box--data">x</div>
        <div class="vae-pipeline-arrow">→</div>
        <div class="vae-pipeline-box vae-pipeline-box--encoder">encoder</div>
        <div class="vae-pipeline-arrow">→</div>
        <div class="vae-pipeline-box vae-pipeline-box--latent">z</div>
        <div class="vae-pipeline-arrow">→</div>
        <div class="vae-pipeline-box vae-pipeline-box--decoder">decoder</div>
        <div class="vae-pipeline-arrow">→</div>
        <div class="vae-pipeline-box vae-pipeline-box--data">x̂</div>
      </div>

      <h1 class="vae-hook__title">Variational Autoencoders</h1>
      <p class="vae-hook__subtitle">Deep generative models, fit by gradient ascent on the ELBO.</p>

      <div class="vae-hook__prose prose">
        <p>Suppose you have a million images of handwritten digits and want a
        probabilistic model that captures the structure of "what digit images look like."
        The model should be able to <strong>generate</strong> new plausible digits,
        <strong>interpolate</strong> between two digits smoothly, and represent each
        digit by a small set of numbers.</p>

        <p>The challenge: the data $x$ lives in a 784-dimensional pixel space. The structure
        is not in any single pixel — it is in patterns spanning many pixels at once. The
        model needs a <strong>latent representation</strong> $z$ in some lower-dimensional
        space that captures the underlying structure, plus a way to map between $z$ and $x$.</p>

        <p>The <strong>variational autoencoder</strong> is one of the elegant answers. It
        trains an <em>encoder</em> that maps $x$ to a distribution over $z$, a
        <em>decoder</em> that maps $z$ back to $x$, and a <em>prior</em> on $z$ that
        structures the latent space — all jointly, by maximizing a single objective:
        the ELBO.</p>
      </div>

      <a class="vae-cta" href="#section-2">Set up the model →</a>
    </div>
    <div id="prereq-strip-container"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountPrereqStrip(sec.querySelector('#prereq-strip-container') as HTMLElement, { prerequisites: meta.prerequisites });
}
