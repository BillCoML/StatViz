import { renderMath, callout, proofToggle } from '@shared/ui';
import { mountJensenGap } from '../viz/jensen-gap';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-3';
  sec.className = 'section';

  const finiteProof = `
    <div class="prose">
      <p>Suppose $X$ takes values $x_1, \\ldots, x_n$ with probabilities $p_1, \\ldots, p_n$
      summing to 1. We want:</p>
      $$\\varphi\\Bigl(\\sum_{i=1}^n p_i x_i\\Bigr) \\;\\leq\\; \\sum_{i=1}^n p_i \\, \\varphi(x_i)$$
      <p>Induction on $n$. The base case $n = 2$ is the definition of convexity with
      $\\lambda = p_1$. For the inductive step, assume the claim for $n-1$ points. Let
      $\\bar p = p_1 + \\cdots + p_{n-1}$ (so $p_n = 1 - \\bar p$). Define
      $\\bar x = \\tfrac{1}{\\bar p} \\sum_{i=1}^{n-1} p_i x_i$. Then by convexity ($n=2$ case
      applied to $\\bar x$ and $x_n$ with weights $\\bar p, 1 - \\bar p$):</p>
      $$\\varphi\\bigl(\\bar p \\, \\bar x + (1 - \\bar p) x_n\\bigr) \\;\\leq\\; \\bar p \\, \\varphi(\\bar x) + (1 - \\bar p) \\, \\varphi(x_n)$$
      <p>The left side is exactly $\\varphi(\\sum_i p_i x_i)$. By the inductive hypothesis
      applied to the renormalized weights $(p_i / \\bar p)_{i=1}^{n-1}$:</p>
      $$\\varphi(\\bar x) \\;\\leq\\; \\sum_{i=1}^{n-1} \\frac{p_i}{\\bar p} \\varphi(x_i)$$
      <p>Substituting and simplifying:</p>
      $$\\varphi\\Bigl(\\sum_i p_i x_i\\Bigr) \\;\\leq\\; \\sum_{i=1}^{n-1} p_i \\, \\varphi(x_i) + (1 - \\bar p) \\varphi(x_n) \\;=\\; \\sum_i p_i \\, \\varphi(x_i) \\quad \\blacksquare$$
    </div>
  `;

  const generalProof = `
    <div class="prose">
      <p><strong>Lemma (supporting line).</strong> For any convex function $\\varphi$ and
      any point $x_0$ in the interior of its domain, there exists a slope $a$ such that</p>
      $$\\varphi(x) \\;\\geq\\; \\varphi(x_0) + a \\, (x - x_0) \\quad \\text{for all } x.$$
      <p>(If $\\varphi$ is differentiable at $x_0$, take $a = \\varphi'(x_0)$. In general,
      $a$ is any element of the subdifferential — guaranteed non-empty for convex functions
      on the interior of their domain.)</p>
      <p>Now apply this with $x_0 = \\mathbb{E}[X]$ and take expectation of both sides over $X$:</p>
      $$\\mathbb{E}[\\varphi(X)] \\;\\geq\\; \\mathbb{E}\\bigl[\\varphi(\\mathbb{E}[X]) + a(X - \\mathbb{E}[X])\\bigr]
        \\;=\\; \\varphi(\\mathbb{E}[X]) + a \\cdot \\mathbb{E}[X - \\mathbb{E}[X]]
        \\;=\\; \\varphi(\\mathbb{E}[X])$$
      <p>The cross-term vanishes because $\\mathbb{E}[X - \\mathbb{E}[X]] = 0$. $\\blacksquare$</p>
      <p>This proof works for any random variable $X$ with $\\mathbb{E}|X| < \\infty$.</p>
    </div>
  `;

  sec.innerHTML = `
    <div class="section-label">§3</div>
    <h2>Jensen's Inequality</h2>
    <div class="prose" id="jensen-statement">
      <h3>Statement</h3>
      <p>Let $\\varphi : \\mathbb{R} \\to \\mathbb{R}$ be a convex function and let $X$ be a
      random variable with $\\mathbb{E}|X| < \\infty$ and $\\mathbb{E}|\\varphi(X)| < \\infty$.
      Then</p>
      $$\\boxed{\\;\\; \\varphi\\bigl(\\mathbb{E}[X]\\bigr) \\;\\leq\\; \\mathbb{E}\\bigl[\\varphi(X)\\bigr] \\;\\;}$$
      <p>If $\\varphi$ is concave, the inequality reverses. If $\\varphi$ is <em>strictly</em>
      convex (resp. concave), equality holds if and only if $X$ is a constant almost surely.</p>
      <p>Read it slowly. It says: <strong>applying a convex function to an average is no
      bigger than averaging the function values</strong>. Visually: replace "average" with
      "midpoint of a chord". The midpoint of a chord lies above the curve (convex). Done.</p>
    </div>
    <div class="worked-example">
      <div class="worked-example-title">Worked example — variance is non-negative</div>
      <p>Take $\\varphi(x) = x^2$ (convex) and any random variable $X$. Jensen gives:</p>
      $$(\\mathbb{E}[X])^2 \\;\\leq\\; \\mathbb{E}[X^2]$$
      <p>Rearranging: $\\mathbb{E}[X^2] - (\\mathbb{E}[X])^2 \\geq 0$, which is exactly
      $\\mathrm{Var}(X) \\geq 0$. <strong>The non-negativity of variance is a one-line
      consequence of Jensen's inequality.</strong></p>
    </div>
    ${proofToggle('Proof for finitely many points (induction)', finiteProof)}
    ${proofToggle('Proof for general random variables (supporting hyperplane)', generalProof, true)}
    ${callout('tip', "The proof you'll actually use", `
      <p>The supporting-hyperplane proof is the one to remember. Every time you need
      Jensen, picture a tangent line to a convex curve at the mean: the curve is above
      the line everywhere; integrate; done.</p>
    `)}
    <div id="viz-jensen" class="viz-wide"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountJensenGap(sec.querySelector('#viz-jensen') as HTMLElement);
}
