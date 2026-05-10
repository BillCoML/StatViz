import { renderMath, crosslinkBack } from '@shared/ui';
import { mountELBOOptimization } from '../viz/elbo-optimization';
import { mountBimodalELBO } from '../viz/bimodal-elbo';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-6';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§6</div>
    <h2>Worked Examples</h2>

    <div class="prose">
      <h3>Act 1 — Conjugate Gaussian (when VI is exact)</h3>
      <p>Take the simplest non-trivial model:</p>
      <ul>
        <li>Prior: $z \\sim \\mathcal{N}(0, \\tau^2)$</li>
        <li>Likelihood: $x_i \\mid z \\sim \\mathcal{N}(z, \\sigma^2)$, $i = 1, \\ldots, n$, conditionally i.i.d.</li>
        <li>Observe: $x_1, \\ldots, x_n$</li>
      </ul>
      <p>Use $\\tau^2 = 1$, $\\sigma^2 = 1$, $n = 3$, data $x = (2.5, 1.7, 3.1)$.
      So $\\bar x = 2.4333$.</p>

      <h4>True posterior (closed form)</h4>
      <p>By conjugacy:</p>
      $$z \\mid x \\;\\sim\\; \\mathcal{N}(\\mu_n, \\sigma_n^2), \\quad
      \\sigma_n^2 = \\left(\\frac{1}{\\tau^2} + \\frac{n}{\\sigma^2}\\right)^{\\!-1} = 0.25, \\quad
      \\mu_n = \\sigma_n^2 \\cdot \\frac{n \\bar x}{\\sigma^2} = 1.825$$
      <p>So the posterior is $\\mathcal{N}(1.825, 0.25)$. The marginal log-evidence is
      $\\log p(x) \\approx -6.1637$.</p>

      <h4>Variational fit</h4>
      <p>Take $\\mathcal{Q} = \\{\\mathcal{N}(\\phi_\\mu, \\phi_\\sigma^2)\\}$. Computing
      the ELBO using Form 2:</p>
      $$\\mathrm{ELBO}(\\phi) \\;=\\; \\sum_{i=1}^{n} \\mathbb{E}_{q_\\phi}[\\log \\mathcal{N}(x_i; Z, \\sigma^2)]
      \\;-\\; D_{\\mathrm{KL}}\\!\\big(\\mathcal{N}(\\phi_\\mu, \\phi_\\sigma^2) \\,\\big\\|\\, \\mathcal{N}(0, \\tau^2)\\big)$$
      <p>The expectation under $Z \\sim \\mathcal{N}(\\phi_\\mu, \\phi_\\sigma^2)$:</p>
      $$\\mathbb{E}_{q_\\phi}[(x_i - Z)^2] \\;=\\; (x_i - \\phi_\\mu)^2 + \\phi_\\sigma^2$$
      <p>So:</p>
      $$\\mathbb{E}_{q_\\phi}[\\log \\mathcal{N}(x_i; Z, \\sigma^2)] \\;=\\;
      -\\tfrac{1}{2}\\log(2\\pi\\sigma^2) \\;-\\;
      \\frac{(x_i - \\phi_\\mu)^2 + \\phi_\\sigma^2}{2\\sigma^2}$$
      <p>The KL between Gaussians (from
      <a href="/lessons/kl-jensen/#kl-gaussians">KL &amp; Jensen §4</a>):</p>
      $$D_{\\mathrm{KL}}\\!\\big(\\mathcal{N}(\\phi_\\mu, \\phi_\\sigma^2) \\,\\big\\|\\, \\mathcal{N}(0, \\tau^2)\\big)
      \\;=\\; \\tfrac{1}{2}\\log \\frac{\\tau^2}{\\phi_\\sigma^2} \\;+\\;
      \\frac{\\phi_\\sigma^2 + \\phi_\\mu^2}{2 \\tau^2} \\;-\\; \\tfrac{1}{2}$$

      <h4>Solving the ELBO</h4>
      <p>Take partial derivatives and set to zero.</p>
      <p>$\\partial \\mathrm{ELBO} / \\partial \\phi_\\mu$:</p>
      $$\\frac{\\partial \\mathrm{ELBO}}{\\partial \\phi_\\mu} \\;=\\;
      \\frac{n(\\bar x - \\phi_\\mu)}{\\sigma^2} \\;-\\; \\frac{\\phi_\\mu}{\\tau^2}
      \\;=\\; 0 \\quad\\Longrightarrow\\quad
      \\phi_\\mu^* \\;=\\; \\frac{n \\bar x / \\sigma^2}{n/\\sigma^2 + 1/\\tau^2} \\;=\\; \\mu_n \\;=\\; 1.825$$
      <p>$\\partial \\mathrm{ELBO} / \\partial \\phi_\\sigma^2$:</p>
      $$\\frac{\\partial \\mathrm{ELBO}}{\\partial \\phi_\\sigma^2} \\;=\\;
      -\\frac{n}{2\\sigma^2} \\;+\\; \\frac{1}{2 \\phi_\\sigma^2} \\;-\\; \\frac{1}{2\\tau^2}
      \\;=\\; 0 \\quad\\Longrightarrow\\quad
      \\phi_\\sigma^{*2} \\;=\\; \\sigma_n^2 \\;=\\; 0.25$$
      <p><strong>The variational optimum exactly matches the posterior</strong>:
      $q^* = \\mathcal{N}(1.825, 0.25) = p(z \\mid x)$. The KL gap is zero.
      ELBO at $\\phi^*$ equals $\\log p(x) = -6.1637$ exactly.</p>
      <p>This is the message of Act 1: <strong>when the variational family contains
      the true posterior, VI is exact</strong>, and the ELBO bound becomes an equality.</p>

      <table>
        <thead>
          <tr><th>$q$</th><th>$\\mathrm{ELBO}(q)$</th><th>KL gap</th></tr>
        </thead>
        <tbody>
          <tr><td>$q^* = \\mathcal{N}(1.825, 0.25)$</td><td>$-6.1637$</td><td>$0.0000$</td></tr>
          <tr><td>$q = $ prior $\\mathcal{N}(0, 1)$</td><td>$-13.6318$</td><td>$7.4681$</td></tr>
          <tr><td>$q = \\mathcal{N}(2.43, 0.33)$ (MLE-ish)</td><td>$-6.9267$</td><td>$0.7630$</td></tr>
        </tbody>
      </table>
    </div>

    <div id="viz-elbo-optimization" class="viz-wide"></div>

    <div class="prose" id="bimodal-failure">
      <h3>Act 2 — Bimodal posterior (when VI fails gracefully)</h3>
      <p>Imagine a model whose posterior comes out to</p>
      $$p(z \\mid x) \\;=\\; \\tfrac{1}{2}\\mathcal{N}(z; -3, 1) \\;+\\; \\tfrac{1}{2}\\mathcal{N}(z; 3, 1)$$
      <p>— bimodal, symmetric. (We constructed this so $\\log p(x) = 0$ exactly: the
      unnormalized joint integrates to 1.)</p>
      <p>The variational family $\\mathcal{Q} = \\{\\mathcal{N}(\\phi_\\mu, \\phi_\\sigma^2)\\}$
      contains only <strong>unimodal</strong> Gaussians. The posterior is bimodal. The family
      does not contain the truth.</p>

      <p>Recognize this distribution — it's the same bimodal target from
      <a href="/lessons/kl-jensen/#reverse-kl">KL &amp; Jensen §7</a>. Back there we computed
      the reverse-KL fit numerically and got $q^{\\text{rev}} \\approx \\mathcal{N}(\\pm 3, 1.05)$.
      <strong>That's exactly the ELBO maximum here.</strong> Maximizing ELBO is minimizing
      reverse KL — same optimization, two names.</p>

      <table>
        <thead>
          <tr><th>$q$</th><th>$\\mathrm{ELBO}(q)$</th><th>KL gap</th></tr>
        </thead>
        <tbody>
          <tr><td>$q^* = \\mathcal{N}(\\pm 3, 1.05)$ (mode-seeking)</td><td>$-0.6888$</td><td>$0.6888$</td></tr>
          <tr><td>$q = \\mathcal{N}(0, 10)$ (forward-KL fit, mass-covering)</td><td>$-0.7789$</td><td>$0.7789$</td></tr>
          <tr><td>$q = \\mathcal{N}(0, 1)$ (centered, narrow)</td><td>$-2.6934$</td><td>$2.6934$</td></tr>
          <tr><td>true posterior $0.5\\mathcal{N}(-3,1) + 0.5\\mathcal{N}(3,1)$ (not in family)</td><td>$0.0000$</td><td>$0.0000$</td></tr>
        </tbody>
      </table>

      <p>Note the KL gap at the optimum is $\\approx \\log 2 = 0.693$ — close to the
      entropy cost of "committing to one mode out of two equal modes." A revealing
      back-of-envelope.</p>
    </div>

    ${crosslinkBack({
      toLesson: 'kl-jensen',
      toAnchor: 'reverse-kl',
      toAnchorLabel: '7',
      body: `<p><strong>The same fit, two derivations.</strong> In KL §7 we found the
        reverse-KL fit by direct minimization. Here we found it by maximizing ELBO.
        By the fundamental identity of §3, those are the same optimization — and indeed
        both give $q^* \\approx \\mathcal{N}(\\pm 3, 1.05)$. VI is reverse-KL minimization
        in formal disguise.</p>`,
    })}

    <div id="viz-bimodal-elbo" class="viz-wide"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountELBOOptimization(sec.querySelector('#viz-elbo-optimization') as HTMLElement);
  mountBimodalELBO(sec.querySelector('#viz-bimodal-elbo') as HTMLElement);
}
