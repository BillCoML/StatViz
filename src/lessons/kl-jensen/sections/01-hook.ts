import { renderMath, mountPrereqStrip } from '@shared/ui';
import { meta } from '../meta';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-1';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="kl-hook">
      <svg class="kl-hook__svg" viewBox="0 0 280 100" aria-hidden="true">
        <path d="M 0 90 C 50 90, 70 10, 110 10 C 150 10, 170 90, 220 90 L 220 90 L 0 90 Z"
              fill="var(--dist-p)" fill-opacity="0.25" stroke="var(--dist-p)" stroke-width="1.5" />
        <path d="M 60 90 C 110 90, 130 10, 170 10 C 210 10, 230 90, 280 90 L 280 90 L 60 90 Z"
              fill="var(--dist-q)" fill-opacity="0.25" stroke="var(--dist-q)" stroke-width="1.5" />
      </svg>
      <h1 class="kl-hook__title">KL Divergence &amp; Jensen's Inequality</h1>
      <div id="prereq-strip"></div>
      <div class="prose">
        <p>You've used Jensen's inequality once or twice in a homework problem.
        You've maybe seen KL divergence in passing — a "distance" between
        distributions, except it isn't a distance.</p>
        <p>What you might not yet appreciate is that <em>almost every modern result
        in probabilistic machine learning leans on these two ideas</em>. The
        convergence proof for EM uses them. The ELBO is built from them. The
        training objective for variational autoencoders is one of them. The
        loss function for diffusion models decomposes into a sum of them.</p>
        <p>This page earns the reader the right to use them. We'll prove
        Jensen's inequality, define KL divergence properly, prove that
        $\\mathrm{KL} \\geq 0$, and end with the picture that explains half of
        variational inference: <strong>forward vs reverse KL</strong>.</p>
      </div>
      <div class="kl-hook__cta">
        <a href="#section-2" class="cta-btn">Let's start with convexity →</a>
      </div>
    </div>
  `;
  container.appendChild(sec);
  mountPrereqStrip(sec.querySelector('#prereq-strip') as HTMLElement, {
    prerequisites: meta.prerequisites,
  });
  renderMath(sec);
}
