import { renderMath, callout } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-5';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§5</div>
    <h2>The VI Algorithm</h2>
    <div class="prose" id="reparam-trick">
      <p>So we want to maximize $\\mathrm{ELBO}(q)$ over a tractable family $\\mathcal{Q}$.
      There are three standard ways.</p>

      <h3>5a. Mean-field VI with closed-form updates (CAVI)</h3>
      <p>Assume $q$ factorizes: $q(z) = \\prod_{i=1}^d q_i(z_i)$. This is the
      <strong>mean-field assumption</strong>. Then the ELBO becomes a sum of pieces,
      and a coordinate-ascent update for each $q_i$ has a closed form:</p>
      $$q_j^*(z_j) \\;\\propto\\; \\exp\\Big(\\mathbb{E}_{q_{-j}}\\!\\big[\\log p(x, z)\\big]\\Big)$$
      <p>where $q_{-j}$ means all the other factors. For models in the exponential family
      with conjugate priors, the update is simple arithmetic over natural parameters.
      (Derivation: take the functional derivative of ELBO with respect to $q_j$ subject
      to $\\int q_j = 1$, set to zero. Sketch only — not the focus of this lesson.)</p>
      <p>CAVI is <strong>exact in its updates</strong> but the mean-field assumption is a
      strong restriction. It breaks correlations between latent dimensions.</p>

      <h3>5b. Gradient-based VI with parametric $q$</h3>
      <p>Parametrize $q$ by $\\phi$, e.g., $q_\\phi(z) = \\mathcal{N}(\\mu_\\phi, \\Sigma_\\phi)$.
      Compute $\\nabla_\\phi \\mathrm{ELBO}$ and ascend.</p>
      <p>The reconstruction term $\\mathbb{E}_{q_\\phi}[\\log p(x \\mid Z)]$ is the awkward
      one — its gradient with respect to $\\phi$ requires differentiating through a sample
      from $q_\\phi$. The standard trick:</p>

      <h3>5c. The reparameterization trick</h3>
      <p>Suppose we can write $Z = g_\\phi(\\epsilon)$ where $\\epsilon$ is drawn from a
      fixed noise distribution $p(\\epsilon)$ that doesn't depend on $\\phi$. (Example: for
      $q_\\phi = \\mathcal{N}(\\mu_\\phi, \\sigma_\\phi^2)$, use
      $g_\\phi(\\epsilon) = \\mu_\\phi + \\sigma_\\phi \\epsilon$ with
      $\\epsilon \\sim \\mathcal{N}(0, 1)$.)</p>
      <p>Then for any function $f$:</p>
      $$\\mathbb{E}_{Z \\sim q_\\phi}[f(Z)] \\;=\\; \\mathbb{E}_{\\epsilon \\sim p(\\epsilon)}\\!\\left[f\\big(g_\\phi(\\epsilon)\\big)\\right]$$
      <p>The right-hand side has $\\phi$ inside the expectand only — so the gradient
      passes inside:</p>
      $$\\nabla_\\phi \\mathbb{E}_{Z \\sim q_\\phi}[f(Z)] \\;=\\;
      \\mathbb{E}_{\\epsilon \\sim p(\\epsilon)}\\!\\left[\\nabla_\\phi f\\big(g_\\phi(\\epsilon)\\big)\\right]$$
      <p>A Monte Carlo estimate is one sample of $\\epsilon$, plug into
      $\\nabla_\\phi f(g_\\phi(\\epsilon))$, done. <strong>This is the trick that makes
      VAEs trainable end-to-end with backpropagation.</strong></p>

      <h3>5d. Black-box VI (REINFORCE / score-function estimators)</h3>
      <p>When $g_\\phi$ doesn't exist (e.g., $z$ is discrete), use the score-function
      gradient $\\nabla_\\phi \\mathbb{E}_{q_\\phi}[f(Z)] = \\mathbb{E}_{q_\\phi}[f(Z) \\nabla_\\phi \\log q_\\phi(Z)]$.
      Higher variance, but unbiased and applicable everywhere.</p>
    </div>

    ${callout('tip', 'What we\'ll demonstrate',
      `<p>The §6 worked examples use <strong>CAVI</strong> for the conjugate Gaussian
      case (closed-form updates) and <strong>gradient ascent</strong> with the
      reparameterization trick for the bimodal case. That's a small but representative
      slice of the VI toolkit.</p>`
    )}
  `;
  container.appendChild(sec);
  renderMath(sec);
}
