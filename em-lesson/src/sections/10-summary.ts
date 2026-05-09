import { renderMath } from '../katex-render';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-10';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§10</div>
    <h2>Summary</h2>
    <div class="summary-card">
      <h3>What you now know</h3>
      <ul class="summary-list">
        <li>The EM algorithm solves maximum-likelihood estimation when data is
        <strong>incomplete</strong> (some variables are hidden).</li>
        <li>The <strong>E-step</strong> computes the expected complete-data
        log-likelihood $Q(\\theta \\mid \\theta^{(t)})$ by computing posterior
        probabilities (responsibilities) over the hidden variables.</li>
        <li>The <strong>M-step</strong> maximizes $Q$ over $\\theta$, yielding
        a closed-form update in many exponential-family models.</li>
        <li>EM is <strong>guaranteed to converge monotonically</strong>: each
        iteration increases the observed-data log-likelihood.</li>
        <li>EM converges to a <strong>local optimum</strong>, not necessarily
        the global one. Use multiple random restarts in practice.</li>
        <li>The two-coin model is an instance of a <strong>Gaussian mixture model</strong>
        (with binary instead of Gaussian observations) — the same EM derivation
        applies to GMMs, HMMs, and many other latent variable models.</li>
      </ul>
    </div>

    <div class="where-next-card">
      <h3>Where to go next</h3>
      <ul class="where-next-list">
        <li>
          <strong>Do &amp; Batzoglou (2008).</strong>
          "What is the expectation maximization algorithm?"
          <em>Nature Biotechnology</em>, 26, 897–899.
          The original exposition this lesson is based on.
        </li>
        <li>
          <strong>Dempster, Laird &amp; Rubin (1977).</strong>
          "Maximum likelihood from incomplete data via the EM algorithm."
          <em>Journal of the Royal Statistical Society B</em>, 39(1), 1–38.
          The seminal paper.
        </li>
        <li>
          <strong>Bishop (2006).</strong>
          <em>Pattern Recognition and Machine Learning</em>, Chapter 9.
          Thorough treatment of EM for mixture models and beyond.
        </li>
        <li>
          <strong>Neal &amp; Hinton (1998).</strong>
          "A view of the EM algorithm that justifies incremental, sparse, and other variants."
          In <em>Learning in Graphical Models</em>. The ELBO perspective in full generality.
        </li>
      </ul>
    </div>
  `;

  container.appendChild(sec);
  renderMath(sec);
}
