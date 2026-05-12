import { renderMath, mountRoadmapMini } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-7';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§7</div>
    <h2>Where You'll See This</h2>
    <div class="prose">
      <p>Four identities. Here's where each shows up in lessons you may
      already have read, or will read next.</p>

      <h3>Outgoing references (these lessons depend on this one)</h3>
      <ul>
        <li><strong>VAE</strong> — uses <a href="#kl-mvn-diag">§3 diagonal KL</a> for the
        regularizer term in the loss; uses <a href="#reparam-matrix">§4 reparameterization</a>
        to backprop through the encoder's sample.</li>
        <li><strong>DDPM</strong> — uses <a href="#kl-mvn">§3 shared-covariance KL</a> for the
        per-timestep loss; uses <a href="#conditioning">§5 conditioning</a> to derive the
        analytical posterior $q(z_{t-1} \\mid z_t, x_0)$;
        uses <a href="#reparam-matrix">§4 reparameterization</a> to sample noise
        along the diffusion chain.</li>
        <li><strong>Score Matching</strong> — uses <a href="#reparam-matrix">§4 reparameterization</a>
        to construct noisy data; uses the score of a Gaussian (which falls out of §2).</li>
      </ul>

      <h3>Incoming references (this lesson depends on these)</h3>
      <ul>
        <li><strong>KL &amp; Jensen</strong> — established the univariate Gaussian KL in §4;
        §3 here is its multivariate generalization. The non-negativity property used
        implicitly here is also from there.</li>
      </ul>

      <h3>Sidebar topics that lean on the same identities</h3>
      <ul>
        <li><strong>Kalman filtering</strong> — repeated linear-Gaussian Bayes (§6).</li>
        <li><strong>Gaussian processes</strong> — conditioning on observations (§5).</li>
        <li><strong>Linear-Gaussian state-space models</strong> — combine §5 and §6.</li>
      </ul>
    </div>

    <div id="roadmap-mini"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountRoadmapMini(sec.querySelector('#roadmap-mini') as HTMLElement, {
    currentLessonId: 'gaussian-cookbook',
  });
}
