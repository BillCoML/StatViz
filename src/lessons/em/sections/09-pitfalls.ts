import { renderMath } from '@shared/ui';
import { callout } from '@shared/ui';
import { mountMultiRestartGallery } from '../viz/multi-restart-gallery';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-9';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§9</div>
    <h2>Pitfalls</h2>
    <div class="prose">
      <p>EM is powerful, but several failure modes deserve attention.</p>
      <ol class="pitfalls-list">
        <li>
          <strong>Local optima.</strong> EM always ascends — but it only finds a
          <em>local</em> maximum. For our two-coin problem the likelihood surface has
          two global maxima that are reflections of each other across the diagonal.
          Starting near $(0.5, 0.5)$ may cause slow convergence or convergence to
          a saddle point where both coins look identical.
        </li>
        <li>
          <strong>Symmetry traps.</strong> If $\\theta_A^{(0)} = \\theta_B^{(0)}$, then
          $\\gamma_i^A = \\gamma_i^B = 0.5$ for all trials. The M-step then sets
          both coins to the same value — the grand average. EM is stuck. Always
          break symmetry with different initial values.
        </li>
        <li>
          <strong>Slow convergence near saddles.</strong> The convergence rate of EM
          is linear (geometric), with rate governed by the fraction of "missing
          information." Near a saddle the effective rate approaches 1 and convergence
          becomes extremely slow.
        </li>
        <li>
          <strong>Unidentifiability.</strong> Our model has a label-switching symmetry:
          swapping $\\theta_A \\leftrightarrow \\theta_B$ gives the same likelihood.
          EM cannot distinguish which mode is "coin A" — it just finds one of them.
          In practice, impose an ordering constraint (e.g., $\\theta_A \\geq \\theta_B$)
          after convergence.
        </li>
        <li>
          <strong>Generalizing to $K$ coins.</strong> With $K$ coins and unequal
          mixing weights $\\pi_1, \\ldots, \\pi_K$, the responsibility formula becomes:
          $$\\gamma_i^k = \\frac{\\pi_k \\cdot (\\theta_k)^{x_i}(1-\\theta_k)^{10-x_i}}{\\sum_{j=1}^K \\pi_j \\cdot (\\theta_j)^{x_i}(1-\\theta_j)^{10-x_i}}$$
          and the M-step for the mixing weights is:
          $$\\pi_k^{(t+1)} = \\frac{1}{5} \\sum_{i=1}^5 \\gamma_i^k$$
          The structure is identical — EM scales gracefully.
          <em>(Note: the implementation in this lesson hard-codes
          $\\pi_A = \\pi_B = \\tfrac{1}{2}$ rather than estimating it via the
          formula above, matching the assumption stated in §2.)</em>
        </li>
      </ol>

      <h3>Practical remedy: multiple random restarts</h3>
      <p>The standard defense against local optima is to run EM from many different
      random starting points and keep the solution with the highest log-likelihood.
      The gallery below shows 30 random restarts, colored by which mode they converge
      to. You can see how initializations on one side of the diagonal tend to find
      one mode, and those on the other side find the other.</p>
    </div>
    ${callout('tip', 'Rule of thumb',
    `<p>For mixture models, run at least $3K$ random restarts where $K$ is the number
    of components. For our two-coin example, that is 6 — though 30 gives a clearer picture.</p>`,
  )}
    <div id="viz-multi-restart" class="viz-wide"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountMultiRestartGallery(sec.querySelector('#viz-multi-restart') as HTMLElement);
}
