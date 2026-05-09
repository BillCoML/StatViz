import { renderMath, crosslinkForward } from '@shared/ui';
import { mountForwardVsReverseFit } from '../viz/fwd-vs-rev-fit';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-7';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§7</div>
    <h2>Forward vs Reverse KL</h2>
    <div class="prose" id="reverse-kl">
      <p>Here's the picture that explains half of variational inference.</p>
      <p>Suppose we have a complicated <strong>target</strong> distribution $p$ and we
      want to <strong>approximate</strong> it with a simpler distribution $q$ chosen from
      some tractable family $\\mathcal{Q}$ (e.g., $q = \\mathcal{N}(\\mu, \\sigma^2)$ with
      $\\mu, \\sigma$ to be chosen). Two natural objectives:</p>
      $$q^{\\text{fwd}} \\;:=\\; \\arg\\min_{q \\in \\mathcal{Q}} \\, D_{\\mathrm{KL}}(p \\,\\|\\, q) \\qquad \\text{(\\"forward KL\\")}$$
      $$q^{\\text{rev}} \\;:=\\; \\arg\\min_{q \\in \\mathcal{Q}} \\, D_{\\mathrm{KL}}(q \\,\\|\\, p) \\qquad \\text{(\\"reverse KL\\")}$$
      <p>These give <em>very different</em> answers. Intuitively:</p>
      <ul>
        <li><strong>Forward KL is mass-covering.</strong> It penalizes $q(x)$ being
        <strong>small</strong> wherever $p(x)$ is <strong>large</strong> (because
        $\\log(p/q)$ blows up there). So $q$ stretches to cover the full support of $p$ —
        it would rather over-extend than miss anything.</li>
        <li><strong>Reverse KL is mode-seeking.</strong> It penalizes $q(x)$ being
        <strong>large</strong> wherever $p(x)$ is <strong>small</strong> (because
        $\\log(q/p)$ blows up there). So $q$ contracts onto a region where $p$ has
        substantial mass — it would rather miss some of $p$'s modes than place mass where
        $p$ has none.</li>
      </ul>
      <p>The cleanest demonstration: <strong>target $p$ is a bimodal mixture, family
      $\\mathcal{Q}$ is unimodal Gaussians</strong>.</p>
    </div>

    <div class="worked-example">
      <div class="worked-example-title">Worked example — bimodal target</div>
      <p>Let $p(x) = \\tfrac{1}{2} \\mathcal{N}(x; -3, 1) + \\tfrac{1}{2} \\mathcal{N}(x; 3, 1)$
      and let $\\mathcal{Q} = \\{ \\mathcal{N}(\\mu, \\sigma^2) \\}$.</p>
      <p><strong>Forward KL minimum.</strong> Minimizing $D(p \\| q)$ over a Gaussian
      family is a <em>moment projection</em>: the KL projection of any distribution
      $p$ onto an exponential family matches the family's sufficient statistics to
      those of $p$. For Gaussians the sufficient statistics are mean and variance,
      so the optimum is $q^{\\text{fwd}} = \\mathcal{N}(\\mathbb{E}_p[X], \\mathrm{Var}_p[X])$.
      For the symmetric mixture, $\\mathbb{E}_p[X] = 0$ and, using
      $\\mathrm{Var}_p[X] = \\mathbb{E}_p[X^2] - (\\mathbb{E}_p[X])^2$ with
      $\\mathbb{E}_p[X^2] = \\tfrac{1}{2}(1 + 9) + \\tfrac{1}{2}(1 + 9) = 10$,
      we get $\\mathrm{Var}_p[X] = 10 - 0 = 10$.
      So $q^{\\text{fwd}} = \\mathcal{N}(0, 10)$ — wide, centered between the two modes,
      mass everywhere $p$ has any.</p>
      <p><strong>Reverse KL minimum.</strong> Numerical optimization (no closed form)
      finds $q^{\\text{rev}} \\approx \\mathcal{N}(\\pm 3, 1.05)$ — by symmetry there are
      two optima, each sitting tightly on one mode and ignoring the other.</p>
    </div>

    <div class="prose">
      <p><strong>Picture this</strong> (referenced from the visualization below):
      forward KL gives a single fat Gaussian straddling both modes; reverse KL gives a
      narrow Gaussian sitting on one mode. <strong>Same target, different objectives,
      completely different fits.</strong></p>
    </div>

    ${crosslinkForward({
      toLesson: 'elbo-vi',
      body: `<p>Variational inference uses <strong>reverse KL</strong>: it picks $q$ to
        minimize $D_{\\mathrm{KL}}(q \\,\\|\\, p_{\\text{posterior}})$. The mode-seeking
        behavior is exactly why VI sometimes underestimates posterior uncertainty — it
        commits to one mode and ignores others. The next lesson will show that minimizing
        reverse KL is equivalent to maximizing the <strong>evidence lower bound (ELBO)</strong>,
        the workhorse objective of modern probabilistic ML.</p>`,
    })}

    <div id="viz-fr" class="viz-wide viz-full"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountForwardVsReverseFit(sec.querySelector('#viz-fr') as HTMLElement);
}
