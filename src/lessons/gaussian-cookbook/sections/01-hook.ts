import { renderMath, mountPrereqStrip } from '@shared/ui';
import { meta } from '../meta';

function jumpToTable(): string {
  return `
    <div class="jump-table">
      <a class="jump-chip" href="#kl-mvn">
        <div class="jump-chip__title">§3 — KL Divergence</div>
        <div class="jump-chip__desc">Closed-form KL between two multivariate Gaussians</div>
      </a>
      <a class="jump-chip" href="#reparam-matrix">
        <div class="jump-chip__title">§4 — Reparameterization</div>
        <div class="jump-chip__desc">Differentiable sampling via $Z = \\mu + L\\varepsilon$</div>
      </a>
      <a class="jump-chip" href="#conditioning">
        <div class="jump-chip__title">§5 — Conditioning</div>
        <div class="jump-chip__desc">$Y \\mid X = x$ from a jointly Gaussian pair</div>
      </a>
      <a class="jump-chip" href="#linear-gauss-bayes">
        <div class="jump-chip__title">§6 — Linear-Gaussian Bayes</div>
        <div class="jump-chip__desc">Gaussian prior + linear Gaussian likelihood → posterior</div>
      </a>
    </div>
  `;
}

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-1';
  sec.className = 'section';
  sec.dataset.anchor = 'intro';
  sec.innerHTML = `
    <div class="gauss-hook">
      <div class="section-label">§1</div>
      <h1 class="gauss-hook__title">Gaussian Cookbook</h1>
      <div id="gauss-prereq-strip"></div>
      <div class="prose" style="text-align:left;">
        <p>Most of the math in VAE, DDPM, and score-based models reduces to
        manipulating Gaussians. Specifically, four identities show up over and over:</p>
        <ul>
          <li>The <strong>KL between two Gaussians</strong> — closed form, used everywhere.</li>
          <li>The <strong>reparameterization trick</strong> — how to backpropagate through
          a Gaussian sample.</li>
          <li><strong>Conditioning</strong> — given a joint Gaussian over $(X, Y)$, what's
          the distribution of $Y$ given $X = x$?</li>
          <li><strong>Linear-Gaussian Bayes</strong> — given a Gaussian prior and a Gaussian
          likelihood with a linear mean, what's the posterior?</li>
        </ul>
        <p>This page collects them. Each is stated, derived, and demonstrated
        with numbers you can verify. The page is designed to be <strong>looked up</strong>
        — a reference, not a story. Jump to whichever identity you need.</p>
      </div>
      ${jumpToTable()}
    </div>
  `;
  container.appendChild(sec);
  mountPrereqStrip(sec.querySelector('#gauss-prereq-strip') as HTMLElement, {
    prerequisites: meta.prerequisites,
  });
  renderMath(sec);
}
