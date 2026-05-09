import { renderMath, mountRoadmapMini } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-8';
  sec.className = 'section';

  sec.innerHTML = `
    <div class="section-label">§8</div>
    <h2>Where You'll See This</h2>
    <div class="prose">
      <p>Two inequalities and one divergence. You'll see them everywhere in
      probabilistic ML; here's a curated list of where, with links into the lessons that
      build on them.</p>

      <h3>Already used (look back)</h3>
      <ul>
        <li><strong>EM convergence theorem.</strong> The proof needed
          $\\mathrm{KL}(k(\\cdot \\mid x, \\theta^{(t)}) \\,\\|\\, k(\\cdot \\mid x, \\theta^{(t+1)})) \\geq 0$.
          That's Gibbs' inequality (§5).
          → <a href="/lessons/em/#monotonicity">Revisit the EM convergence proof</a></li>
      </ul>

      <h3>Coming next</h3>
      <ul>
        <li><strong>ELBO &amp; Variational Inference.</strong> Built directly from a
        reverse-KL objective (§7). The "evidence lower bound" is what you get when you
        rearrange $D_{\\mathrm{KL}}(q \\,\\|\\, p_{\\text{posterior}})$.
        → Next lesson on the roadmap.</li>
        <li><strong>Variational Autoencoders.</strong> Loss = reconstruction term + KL
        regularizer. The KL is between the encoder $q(z \\mid x)$ and a prior
        $p(z) = \\mathcal{N}(0, I)$, evaluated using the Gaussian–Gaussian closed form
        derived in §4.</li>
        <li><strong>DDPM (the destination).</strong> The training objective is a sum of
        KL terms, each between two Gaussians along a Markov chain. The chain rule of KL
        (§6) breaks it into per-step pieces; the Gaussian KL formula (§4) makes each
        piece tractable.</li>
      </ul>

      <h3>Adjacent ideas (sidebar lessons)</h3>
      <ul>
        <li><strong>Cross-entropy in classification.</strong> Cross-entropy is just
        $H(p) + D_{\\mathrm{KL}}(p \\,\\|\\, q)$; minimizing cross-entropy = minimizing
        forward KL.</li>
        <li><strong>Maximum likelihood estimation.</strong> MLE minimizes
        $D_{\\mathrm{KL}}(\\widehat p_{\\text{data}} \\,\\|\\, p_\\theta)$ — the forward KL
        between the empirical and model distributions. Every ML textbook says this in
        passing; we now know exactly what it means.</li>
      </ul>
    </div>

    <div id="roadmap-mini"></div>
  `;

  container.appendChild(sec);
  renderMath(sec);
  mountRoadmapMini(sec.querySelector('#roadmap-mini') as HTMLElement, {
    currentLessonId: 'kl-jensen',
  });
}
