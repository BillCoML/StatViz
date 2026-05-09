import { renderMath } from '../katex-render';
import { callout } from '../ui/callout';
import { mountBinomialMixture } from '../viz/binomial-mixture';
import { TRIALS } from '../em/data';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-2';
  sec.className = 'section';

  // Build table rows from TRIALS data
  const tableRows = TRIALS.map(t =>
    `<tr>
      <td>${t.id}</td>
      <td class="sequence">${t.sequence.split('').join(' ')}</td>
      <td>${t.heads}</td>
      <td>${t.tails}</td>
    </tr>`,
  ).join('');

  sec.innerHTML = `
    <div class="section-label">§2</div>
    <h2>The Problem</h2>
    <div class="prose">
      <p>We have five trials. In each trial, one of two coins — let's call them
      coin A and coin B — was selected at random (each with probability $\\frac{1}{2}$),
      then flipped ten times. We observe the flip results but <em>not</em> which
      coin was used.</p>
      <p>Let $\\theta_A \\in (0,1)$ be the probability of heads for coin A, and
      $\\theta_B \\in (0,1)$ for coin B. We write $\\theta = (\\theta_A, \\theta_B)$.
      Our goal is to estimate $\\theta$ from the observed data.</p>
    </div>
    <table class="data-table">
      <thead>
        <tr>
          <th>Trial $i$</th>
          <th>Sequence</th>
          <th>Heads $x_i$</th>
          <th>Tails $10 - x_i$</th>
        </tr>
      </thead>
      <tbody>
        ${tableRows}
      </tbody>
    </table>
    <div class="prose">
      <p>Squint at the table: trials 2 and 3 have lots of heads (9 and 8), suggesting
      one coin is heavily biased toward heads. Trials 1 and 4 have near-even splits
      (5 and 4), consistent with a coin near $\\theta = 0.5$. Trial 5 sits in between
      at 7 heads.</p>
      <p>If we knew which coin produced each trial, computing the maximum-likelihood
      estimate would be trivial — just pool the heads and tails for each coin separately.
      The catch is that we <em>don't</em> know. This is the incomplete-data problem.</p>
    </div>
    <div id="viz-binomial-mixture" class="viz-wide"></div>
    ${callout('info', 'Why this is hard',
    `<p>Because the coin identity $z_i$ is hidden, the likelihood function for $\\theta$
    involves a <em>sum inside a logarithm</em>. Unlike a simple binomial, this has
    no closed-form maximizer. We need a smarter approach — and that's where EM comes in.</p>`,
  )}
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountBinomialMixture(sec.querySelector('#viz-binomial-mixture') as HTMLElement);
}
