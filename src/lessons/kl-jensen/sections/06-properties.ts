import { renderMath } from '@shared/ui';
import { mountBernoulliHeatmap } from '../viz/bernoulli-heatmap';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-6';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§6</div>
    <h2>Properties of KL</h2>
    <div class="prose">
      <p>KL has a handful of properties that get used constantly. We collect them here.</p>

      <h3>1. Asymmetry</h3>
      <p>$D_{\\mathrm{KL}}(p \\,\\|\\, q) \\neq D_{\\mathrm{KL}}(q \\,\\|\\, p)$ in general.
      Worked examples:</p>
      <table class="data-table">
        <thead><tr><th>Comparison</th><th>$D(p \\| q)$</th><th>$D(q \\| p)$</th></tr></thead>
        <tbody>
          <tr><td>$\\mathrm{Bern}(0.7) \\,\\|\\, \\mathrm{Bern}(0.5)$</td><td>0.0823</td><td>0.0872</td></tr>
          <tr><td>$\\mathrm{Bern}(0.9) \\,\\|\\, \\mathrm{Bern}(0.5)$</td><td>0.3681</td><td>0.5108</td></tr>
          <tr><td>$\\mathcal{N}(0, 1) \\,\\|\\, \\mathcal{N}(0, 4)$</td><td>0.318</td><td>0.807</td></tr>
        </tbody>
      </table>
      <p>Notice the third row: when $q$ is much wider than $p$,
      $D(p \\| q) < D(q \\| p)$. Why? Because $D(q \\| p)$ has $q$'s broad mass landing in
      $p$'s tails, where $\\log(q/p)$ is large; whereas $D(p \\| q)$ has $p$'s narrow mass
      on a region where $q$ is reasonably accurate.</p>

      <h3>2. Not a metric</h3>
      <p>Beyond asymmetry, KL also fails the triangle inequality. Counterexamples are
      easy to construct on three Bernoullis. So calling it a "distance" is a useful
      intuition pump but a formal lie.</p>

      <h3>3. Convex in the pair</h3>
      <p>$(p, q) \\mapsto D_{\\mathrm{KL}}(p \\,\\|\\, q)$ is jointly convex. Specifically,
      for any $\\lambda \\in [0, 1]$ and pairs $(p_1, q_1), (p_2, q_2)$,</p>
      $$D_{\\mathrm{KL}}\\!\\bigl(\\lambda p_1 + (1-\\lambda) p_2 \\,\\big\\|\\, \\lambda q_1 + (1-\\lambda) q_2\\bigr)
        \\;\\leq\\; \\lambda \\, D_{\\mathrm{KL}}(p_1 \\,\\|\\, q_1) + (1-\\lambda) \\, D_{\\mathrm{KL}}(p_2 \\,\\|\\, q_2)$$
      <p>(Proof: log-sum inequality. We won't reproduce it here — it's tedious — but
      it's worth knowing the name.)</p>

      <h3>4. Pinsker's inequality</h3>
      <p>KL upper-bounds total variation:</p>
      $$\\| p - q \\|_{\\mathrm{TV}} \\;\\leq\\; \\sqrt{\\tfrac{1}{2} D_{\\mathrm{KL}}(p \\,\\|\\, q)}$$
      <p>So when KL is small, $p$ and $q$ are <em>close</em> in total variation distance —
      they assign similar probabilities to every event. Pinsker is what makes KL useful
      as a <em>practical</em> distance even though it isn't formally one.</p>

      <h3>5. Chain rule</h3>
      <p>For joint distributions $p(x, y) = p(x) p(y \\mid x)$ and analogously for $q$,</p>
      $$D_{\\mathrm{KL}}(p(x, y) \\,\\|\\, q(x, y))
        \\;=\\; D_{\\mathrm{KL}}(p(x) \\,\\|\\, q(x))
        \\;+\\; \\mathbb{E}_{X \\sim p}\\!\\left[D_{\\mathrm{KL}}\\!\\bigl(p(y \\mid X) \\,\\|\\, q(y \\mid X)\\bigr)\\right]$$
      <p>This is the analogue of the chain rule for entropy. We'll use it in the DDPM
      lesson, where the KL across a Markov chain decomposes step by step.</p>
    </div>

    <div id="viz-bern-heatmap" class="viz-wide"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountBernoulliHeatmap(sec.querySelector('#viz-bern-heatmap') as HTMLElement);
}
