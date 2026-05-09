import { renderMath, callout } from '@shared/ui';
import { mountConvexExplorer } from '../viz/convex-explorer';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-2';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§2</div>
    <h2>Convex and Concave Functions</h2>
    <div class="prose">
      <p>Before Jensen's inequality, we need to be precise about what <em>convex</em>
      means. A function $\\varphi : \\mathbb{R} \\to \\mathbb{R}$ is <strong>convex</strong>
      on an interval $I$ if for every $x_1, x_2 \\in I$ and every $\\lambda \\in [0, 1]$,</p>
      $$\\varphi\\bigl(\\lambda x_1 + (1-\\lambda) x_2\\bigr) \\;\\leq\\; \\lambda \\, \\varphi(x_1) + (1-\\lambda) \\, \\varphi(x_2)$$
      <p>Geometrically: the <strong>chord</strong> between any two points on the graph lies
      on or above the curve. The function is <strong>strictly convex</strong> if the
      inequality is strict whenever $x_1 \\neq x_2$ and $\\lambda \\in (0, 1)$ — i.e., the
      chord lies <em>strictly</em> above the curve except at the endpoints.</p>
      <p>A function is <strong>concave</strong> if $-\\varphi$ is convex; equivalently, the
      chord lies on or below the curve.</p>
      <p>Standard examples we'll use throughout the page:</p>
      <table class="data-table">
        <thead><tr><th>Function</th><th>Domain</th><th>Convex / concave</th></tr></thead>
        <tbody>
          <tr><td>$\\varphi(x) = x^2$</td><td>$\\mathbb{R}$</td><td>strictly convex</td></tr>
          <tr><td>$\\varphi(x) = e^x$</td><td>$\\mathbb{R}$</td><td>strictly convex</td></tr>
          <tr><td>$\\varphi(x) = -\\log(x)$</td><td>$(0, \\infty)$</td><td>strictly convex</td></tr>
          <tr><td>$\\varphi(x) = \\log(x)$</td><td>$(0, \\infty)$</td><td>strictly <strong>concave</strong></td></tr>
          <tr><td>$\\varphi(x) = |x|$</td><td>$\\mathbb{R}$</td><td>convex (not strictly)</td></tr>
        </tbody>
      </table>
      <p><strong>Why we care about strict convexity</strong>: it gives us <em>equality
      conditions</em> later. Jensen's inequality with a strictly convex $\\varphi$ becomes
      an equality only when the random variable is degenerate (a constant). This will let
      us prove uniqueness results.</p>
      <p>If $\\varphi$ is twice-differentiable, the test is simple: $\\varphi$ is convex
      iff $\\varphi'' \\geq 0$ everywhere; strictly convex iff $\\varphi'' > 0$ everywhere.</p>
    </div>
    ${callout('tip', 'log is the one to remember', `
      <p>Throughout this page, the function we'll apply Jensen's inequality to is $\\log$
      (concave) — or equivalently $-\\log$ (convex). Every KL property in this page reduces
      to: <em>carefully apply Jensen's inequality to a logarithm</em>. Internalize that and
      the rest is bookkeeping.</p>
    `)}
    <div id="viz-convex" class="viz-wide"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountConvexExplorer(sec.querySelector('#viz-convex') as HTMLElement);
}
