import { renderMath } from '@shared/ui';
import { mountKLCalculator } from '../viz/kl-calculator';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-4';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§4</div>
    <h2>KL Divergence: Definition</h2>
    <div class="prose">
      <h3>Definition</h3>
      <p>Let $p$ and $q$ be probability distributions on the same space. The
      <strong>Kullback–Leibler divergence</strong> of $q$ from $p$ is</p>
      $$\\boxed{\\;\\; D_{\\mathrm{KL}}(p \\,\\|\\, q) \\;:=\\; \\mathbb{E}_{X \\sim p}\\!\\left[\\log \\frac{p(X)}{q(X)}\\right] \\;\\;}$$
      <p>In the discrete case:</p>
      $$D_{\\mathrm{KL}}(p \\,\\|\\, q) \\;=\\; \\sum_x p(x) \\log \\frac{p(x)}{q(x)}$$
      <p>In the continuous case (with densities):</p>
      $$D_{\\mathrm{KL}}(p \\,\\|\\, q) \\;=\\; \\int p(x) \\log \\frac{p(x)}{q(x)} \\, dx$$

      <h3>Conventions and edge cases</h3>
      <p>Two annoying edge cases need conventions:</p>
      <ol>
        <li>$0 \\log 0 = 0$ (justified by $\\lim_{t \\to 0^+} t \\log t = 0$). So points
        where $p(x) = 0$ contribute nothing to the sum.</li>
        <li>If there exists $x$ with $p(x) > 0$ and $q(x) = 0$, then
        $D_{\\mathrm{KL}}(p \\,\\|\\, q) = +\\infty$. This is more than just a convention —
        it captures a <em>real fact</em>. If $q$ assigns zero probability to something
        $p$ thinks is possible, no amount of sampling from $q$ will ever produce that
        event, so the "code based on $q$" cannot describe it. Infinity is correct.</li>
      </ol>
      <p>Formally: $D_{\\mathrm{KL}}(p \\,\\|\\, q) < \\infty$ requires $p$ to be
      <strong>absolutely continuous with respect to</strong> $q$ — written $p \\ll q$.</p>

      <h3>Intuition: extra description length</h3>
      <p>The cleanest interpretation comes from information theory. Suppose data is
      generated according to $p$, but we encode it using a code optimized for $q$
      (assigning $-\\log_2 q(x)$ bits to outcome $x$). The expected description length is
      $-\\mathbb{E}_p[\\log_2 q(X)]$. The <em>minimum possible</em> description length,
      achieved by the code optimized for $p$ itself, is
      $H(p) := -\\mathbb{E}_p[\\log_2 p(X)]$ (the entropy of $p$). The difference is:</p>
      $$\\underbrace{-\\mathbb{E}_p[\\log_2 q(X)]}_{\\text{using $q$-code}}
        \\;-\\; \\underbrace{-\\mathbb{E}_p[\\log_2 p(X)]}_{\\text{using $p$-code (optimal)}}
        \\;=\\; \\mathbb{E}_p\\!\\left[\\log_2 \\frac{p(X)}{q(X)}\\right]
        \\;=\\; D_{\\mathrm{KL}}(p \\,\\|\\, q)$$
      <p>So $D_{\\mathrm{KL}}(p \\,\\|\\, q)$ is <strong>the expected number of extra bits
      required to describe $p$-distributed data using a $q$-optimized code</strong>.
      (In machine learning we usually use natural log, so the unit is "nats" rather than
      "bits" — but the interpretation is identical.)</p>
    </div>

    <div class="worked-example">
      <div class="worked-example-title">Worked example — Bernoullis</div>
      <p>Let $p = \\mathrm{Bern}(0.7)$ and $q = \\mathrm{Bern}(0.5)$. Then</p>
      $$D_{\\mathrm{KL}}(p \\,\\|\\, q) \\;=\\; 0.7 \\log \\tfrac{0.7}{0.5} + 0.3 \\log \\tfrac{0.3}{0.5}
        \\;\\approx\\; 0.7 \\cdot 0.336 + 0.3 \\cdot (-0.511)
        \\;\\approx\\; 0.0823 \\text{ nats}$$
      <p>Reverse direction:</p>
      $$D_{\\mathrm{KL}}(q \\,\\|\\, p) \\;=\\; 0.5 \\log \\tfrac{0.5}{0.7} + 0.5 \\log \\tfrac{0.5}{0.3}
        \\;\\approx\\; 0.0872 \\text{ nats}$$
      <p>Different! Already we see KL is <strong>not symmetric</strong>. We'll come back
      to this in §6.</p>
    </div>

    <div class="worked-example" id="kl-gaussians">
      <div class="worked-example-title">Worked example — Gaussians (full derivation)</div>
      <p>Let $p = \\mathcal{N}(\\mu_1, \\sigma_1^2)$ and $q = \\mathcal{N}(\\mu_2, \\sigma_2^2)$.
      We derive the closed form because it's the building block of every Gaussian-latent
      variational objective (and hence of DDPM).</p>
      <p>Recall $\\log \\mathcal{N}(x; \\mu, \\sigma^2) = -\\tfrac{1}{2} \\log(2\\pi\\sigma^2)
      - \\tfrac{(x - \\mu)^2}{2\\sigma^2}$. So:</p>
      $$\\log \\frac{p(x)}{q(x)} \\;=\\; \\log \\frac{\\sigma_2}{\\sigma_1}
        \\;-\\; \\frac{(x - \\mu_1)^2}{2\\sigma_1^2}
        \\;+\\; \\frac{(x - \\mu_2)^2}{2\\sigma_2^2}$$
      <p>Take expectation under $X \\sim p$. We need
      $\\mathbb{E}_p[(X - \\mu_1)^2] = \\sigma_1^2$ (just the variance) and
      $\\mathbb{E}_p[(X - \\mu_2)^2] = \\sigma_1^2 + (\\mu_1 - \\mu_2)^2$ (shift the mean
      and use the bias-variance decomposition). Substituting:</p>
      $$D_{\\mathrm{KL}}(p \\,\\|\\, q) \\;=\\; \\log\\tfrac{\\sigma_2}{\\sigma_1}
        \\;-\\; \\tfrac{\\sigma_1^2}{2\\sigma_1^2}
        \\;+\\; \\tfrac{\\sigma_1^2 + (\\mu_1 - \\mu_2)^2}{2\\sigma_2^2}$$
      <p>Simplifying:</p>
      $$\\boxed{\\;\\; D_{\\mathrm{KL}}\\!\\bigl(\\mathcal{N}(\\mu_1, \\sigma_1^2) \\,\\big\\|\\, \\mathcal{N}(\\mu_2, \\sigma_2^2)\\bigr) \\;=\\; \\log \\tfrac{\\sigma_2}{\\sigma_1} \\;+\\; \\tfrac{\\sigma_1^2 + (\\mu_1 - \\mu_2)^2}{2 \\sigma_2^2} \\;-\\; \\tfrac{1}{2} \\;\\;}$$
      <p><strong>Sanity checks</strong>: When $\\mu_1 = \\mu_2$ and $\\sigma_1 = \\sigma_2$,
      all three terms vanish — KL is zero, as it should be. When
      $\\mu_1 = 0, \\mu_2 = 1, \\sigma_1 = \\sigma_2 = 1$:
      $\\log 1 + (1 + 1)/2 - 1/2 = \\tfrac{1}{2}$.</p>
    </div>

    <div id="viz-kl-calculator" class="viz-wide"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountKLCalculator(sec.querySelector('#viz-kl-calculator') as HTMLElement);
}
