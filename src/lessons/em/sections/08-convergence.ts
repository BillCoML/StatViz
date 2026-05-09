import { renderMath } from '@shared/ui';
import { callout } from '@shared/ui';
import { proofToggle } from '@shared/ui';
import { crosslinkBack } from '@shared/ui';
import { mountMonotonicityDemo } from '../viz/monotonicity-demo';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-8';
  sec.className = 'section';

  const proofHtml = `
    <div class="prose">
      <p>Use the conditional density of the missing data:</p>
      $$k(z \\mid x, \\theta) \\;:=\\; P(Z = z \\mid x, \\theta) \\;=\\; \\frac{P(x, z \\mid \\theta)}{P(x \\mid \\theta)}$$

      <p>Take the log on both sides of the rearranged identity
      $P(x, z \\mid \\theta) = P(x \\mid \\theta) \\cdot k(z \\mid x, \\theta)$:</p>
      $$\\log P(x, z \\mid \\theta) \\;=\\; \\log P(x \\mid \\theta) \\;+\\; \\log k(z \\mid x, \\theta)$$

      <p>Equivalently, $\\ell_c(\\theta \\mid x, z) = \\ell(\\theta \\mid x) + \\log k(z \\mid x, \\theta)$.
      Solving for $\\ell(\\theta \\mid x)$ (which doesn't depend on $z$):</p>
      $$\\ell(\\theta \\mid x) \\;=\\; \\ell_c(\\theta \\mid x, z) \\;-\\; \\log k(z \\mid x, \\theta)$$

      <p>Now take the expectation of <em>both sides</em> under $z \\sim k(\\cdot \\mid x, \\theta^{(t)})$.
      The left side is unchanged (it doesn't depend on $z$):</p>
      $$\\ell(\\theta \\mid x) \\;=\\;
        \\underbrace{\\mathbb{E}_{Z \\sim k(\\cdot \\mid x, \\theta^{(t)})} [\\ell_c(\\theta \\mid x, Z)]}_{=\\, Q(\\theta \\mid \\theta^{(t)})}
        \\;-\\;
        \\underbrace{\\mathbb{E}_{Z \\sim k(\\cdot \\mid x, \\theta^{(t)})} [\\log k(Z \\mid x, \\theta)]}_{=:\\, H(\\theta \\mid \\theta^{(t)})}$$

      <p>So we have the <strong>decomposition identity</strong>:</p>
      $$\\ell(\\theta \\mid x) \\;=\\; Q(\\theta \\mid \\theta^{(t)}) \\;-\\; H(\\theta \\mid \\theta^{(t)})$$

      <p>Therefore the change in log-likelihood between iterations is:</p>
      $$\\ell(\\theta^{(t+1)} \\mid x) - \\ell(\\theta^{(t)} \\mid x)
        = \\underbrace{\\bigl[ Q(\\theta^{(t+1)} \\mid \\theta^{(t)}) - Q(\\theta^{(t)} \\mid \\theta^{(t)}) \\bigr]}_{\\text{(A)}}
        - \\underbrace{\\bigl[ H(\\theta^{(t+1)} \\mid \\theta^{(t)}) - H(\\theta^{(t)} \\mid \\theta^{(t)}) \\bigr]}_{\\text{(B)}}$$

      <p>We show <strong>(A) is non-negative</strong> and <strong>(B) is non-positive</strong> — making the total non-negative.</p>

      <p><strong>Term (A) $\\geq 0$.</strong>&nbsp; This is immediate from the M-step's definition:
      $\\theta^{(t+1)}$ is <em>defined</em> to maximize $Q(\\theta \\mid \\theta^{(t)})$ over $\\theta$.
      So in particular $Q(\\theta^{(t+1)} \\mid \\theta^{(t)}) \\geq Q(\\theta^{(t)} \\mid \\theta^{(t)})$.</p>

      <p><strong>Term (B) $\\leq 0$.</strong>&nbsp; This is <strong>Gibbs' inequality</strong> (equivalently,
      non-negativity of KL divergence). For any two distributions $p, q$ on the same space,</p>
      $$\\mathbb{E}_{Z \\sim p}[\\log p(Z)] \\;\\geq\\; \\mathbb{E}_{Z \\sim p}[\\log q(Z)]$$
      <p>with equality iff $p = q$. Apply this with
      $p = k(\\cdot \\mid x, \\theta^{(t)})$ and $q = k(\\cdot \\mid x, \\theta^{(t+1)})$:</p>
      $$H(\\theta^{(t)} \\mid \\theta^{(t)})
        = \\mathbb{E}_{Z \\sim k(\\cdot \\mid x, \\theta^{(t)})}[\\log k(Z \\mid x, \\theta^{(t)})]
        \\;\\geq\\; \\mathbb{E}_{Z \\sim k(\\cdot \\mid x, \\theta^{(t)})}[\\log k(Z \\mid x, \\theta^{(t+1)})]
        = H(\\theta^{(t+1)} \\mid \\theta^{(t)})$$
      <p>So $H(\\theta^{(t+1)} \\mid \\theta^{(t)}) - H(\\theta^{(t)} \\mid \\theta^{(t)}) \\leq 0$,
      i.e. term (B) is non-positive. Equivalently:</p>
      $$H(\\theta^{(t)} \\mid \\theta^{(t)}) - H(\\theta^{(t+1)} \\mid \\theta^{(t)})
        = \\mathrm{KL}\\bigl(k(\\cdot \\mid x, \\theta^{(t)}) \\,\\big\\|\\, k(\\cdot \\mid x, \\theta^{(t+1)})\\bigr) \\;\\geq\\; 0$$

      <p>Combining:</p>
      $$\\ell(\\theta^{(t+1)} \\mid x) - \\ell(\\theta^{(t)} \\mid x)
        = \\underbrace{[Q\\text{-gap}]}_{\\geq\\, 0}
        + \\underbrace{\\mathrm{KL}\\bigl(k(\\cdot \\mid x, \\theta^{(t)}) \\big\\| k(\\cdot \\mid x, \\theta^{(t+1)})\\bigr)}_{\\geq\\, 0}
        \\;\\geq\\; 0. \\quad \\blacksquare$$
    </div>
  `;

  sec.innerHTML = `
    <div class="section-label">§8</div>
    <h2>Why It Works</h2>
    <div class="prose" id="monotonicity">
      <h3>The Monotonicity Theorem</h3>
      <p>Every EM iteration is guaranteed to increase (or not decrease) the
      observed-data log-likelihood:</p>
      $$\\ell(\\theta^{(t+1)} \\mid x) \\;\\geq\\; \\ell(\\theta^{(t)} \\mid x) \\quad \\text{for all } t \\geq 0$$
      <p>This is the fundamental convergence guarantee of EM. It means the algorithm
      can never make things worse, and the log-likelihood sequence is monotone
      non-decreasing. Since the log-likelihood is bounded above (probabilities are
      $\\leq 1$), the <em>sequence of log-likelihood values</em> must converge.</p>
      <p><em>Caveat:</em> convergence of the log-likelihood values does <strong>not</strong>
      automatically imply convergence of the parameter sequence
      $\\{\\theta^{(t)}\\}$ — the iterates can in principle wander on a level set
      of $\\ell$. Parameter convergence requires additional regularity (see Wu, 1983).
      For our two-coin model the regularity holds and the parameters do converge.</p>
    </div>
    ${proofToggle('Proof of monotonicity (click to expand)', proofHtml)}
    ${crosslinkBack({
      toLesson: 'kl-jensen',
      toAnchor: 'gibbs-inequality',
      toAnchorLabel: '5',
      body: `<p>The proof above leans on
        $\\mathrm{KL}\\bigl(k(\\cdot \\mid x, \\theta^{(t)}) \\,\\big\\|\\, k(\\cdot \\mid x, \\theta^{(t+1)})\\bigr) \\geq 0$
        — the non-negativity of KL divergence, also known as Gibbs' inequality.
        We took it on trust here. The KL &amp; Jensen lesson proves it from first principles
        using Jensen's inequality on $\\log$.</p>`,
    })}
    <div class="prose">
      <p>The demo below visualizes the two components of the log-likelihood gain
      at each EM step: the Q-function improvement (always $\\geq 0$) and the
      KL correction (always $\\leq 0$ in magnitude). Their sum is the net gain,
      always non-negative.</p>
    </div>
    <div id="viz-monotonicity" class="viz-wide"></div>
    ${callout('warning', 'What EM does NOT guarantee',
    `<p>Monotonicity guarantees convergence to a <em>local</em> maximum or saddle point —
    not necessarily the <em>global</em> maximum. For our two-coin problem, EM reliably
    finds one of the two symmetric global maxima (at $(0.797, 0.520)$ or $(0.520, 0.797)$)
    depending on initialization. But in more complex models, EM can get stuck. Section §9
    covers practical strategies.</p>`,
  )}
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountMonotonicityDemo(sec.querySelector('#viz-monotonicity') as HTMLElement);
}
