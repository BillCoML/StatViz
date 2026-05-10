import { renderMath, mountPrereqStrip } from '@shared/ui';
import { meta } from '../meta';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-1';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="elbo-hook">
      <h1 class="elbo-hook__title">ELBO &amp; Variational Inference</h1>
      <div id="prereq-strip"></div>
      <div class="prose">
        <p>The integral $p(x) = \\int p(x, z) \\, dz$ that defines the marginal
        likelihood — the <em>evidence</em> — is the central quantity in
        probabilistic modeling. It's also, for nearly every model worth
        caring about, <strong>intractable</strong>.</p>
        <p>Variational inference replaces this intractable computation with an
        optimization problem. We pick a tractable family of distributions
        $q$, find one that's "close" to the true posterior, and use it as
        a stand-in. The objective we maximize is called the <strong>Evidence
        Lower Bound</strong> — ELBO for short — and it's a lower bound on
        $\\log p(x)$ that's tight when $q$ matches the posterior exactly.</p>
        <p>By the end of this lesson you'll have the fundamental identity in
        your back pocket, you'll know what VI does and what it can't do,
        and you'll see something nice: <strong>the EM algorithm is exactly
        coordinate ascent on the ELBO</strong>.</p>
      </div>
      <div class="elbo-hook__cta">
        <a href="#section-2" class="cta-btn">Set up the problem →</a>
      </div>
    </div>
  `;
  container.appendChild(sec);
  mountPrereqStrip(sec.querySelector('#prereq-strip') as HTMLElement, {
    prerequisites: meta.prerequisites,
  });
  renderMath(sec);
}
