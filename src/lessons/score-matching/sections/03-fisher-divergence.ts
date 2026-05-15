import { renderMath, crosslinkSidebar } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-3';
  sec.className = 'section';
  sec.setAttribute('data-anchor', 'fisher-divergence');
  sec.innerHTML = `
    <div class="section-label">§3</div>
    <h2>Fisher Divergence and the Score Matching Objective</h2>
    <div class="prose">
      <p>To <strong>fit</strong> a score model $s_\\theta(x)$ to data, we need a loss.
      The natural one: penalize the squared difference between the model's score and the
      data's true score.</p>

      <div class="formula-box">
        $$\\mathcal{L}_{\\mathrm{SM}}(\\theta) = \\mathbb{E}_{x \\sim p_{\\mathrm{data}}}
        \\!\\left[\\|s_\\theta(x) - \\nabla_x \\log p_{\\mathrm{data}}(x)\\|^2\\right]$$
      </div>

      <p>This quantity — the expected squared $L^2$ distance between the two score fields —
      is the <strong>Fisher divergence</strong>
      $\\mathcal{F}(p_{\\mathrm{data}} \\,\\|\\, p_\\theta)$, sometimes called the
      <strong>relative Fisher information</strong>.</p>

      <p>Properties:</p>
      <ul>
        <li>Non-negative: $\\mathcal{F} \\geq 0$, with equality iff
        $s_\\theta(x) = \\nabla \\log p_{\\mathrm{data}}(x)$ for
        $p_{\\mathrm{data}}$-almost-every $x$.</li>
        <li>Asymmetric in $p_{\\mathrm{data}}$ and $p_\\theta$ — like KL but not the
        same quantity.</li>
        <li>Connected to KL via De Bruijn's identity:
        $\\partial_t D_{\\mathrm{KL}}(p * \\mathcal{N}_t \\,\\|\\, q * \\mathcal{N}_t)
        = -\\tfrac{1}{2}\\mathcal{F}(p * \\mathcal{N}_t,\\, q * \\mathcal{N}_t)$
        where $p * \\mathcal{N}_t$ is $p$ convolved with Gaussian noise of variance $tI$.
        Fisher divergence is KL's "derivative through noise smoothing."</li>
      </ul>

      <h3>The fundamental problem</h3>
      <p>The loss $\\mathcal{L}_{\\mathrm{SM}}$ requires
      $\\nabla_x \\log p_{\\mathrm{data}}(x)$ — the score of the data distribution —
      which we don't have. We only have samples from $p_{\\mathrm{data}}$, not its
      score field.</p>

      <p>Three workarounds, each removing the dependence on
      $\\nabla \\log p_{\\mathrm{data}}$ in a different way:</p>
      <ol>
        <li><strong>Implicit score matching</strong> (Hyvärinen 2005) — integration by
        parts. Trades $\\nabla \\log p_{\\mathrm{data}}$ for a trace of the Jacobian
        of $s_\\theta$. <strong>§4.</strong></li>
        <li><strong>Denoising score matching</strong> (Vincent 2011) — corrupt the data
        with noise and match the score of the noise-perturbed distribution, which has a
        tractable conditional. <strong>§5.</strong></li>
        <li><strong>Sliced score matching</strong> (Song et al. 2019) — project onto
        random directions, avoid the full Jacobian. Mentioned briefly; not the focus.</li>
      </ol>

      <p>§4 develops route 1. §5 develops route 2 — the one that DDPM uses.</p>

      <div class="sm-conceptual-diagram" aria-label="Fisher divergence diagram">
        <div class="sm-two-fields">
          <div class="sm-field sm-field--data">data score $s_\\text{data}$</div>
          <div class="sm-field-gap">← gap = $\\mathcal{F}$ →</div>
          <div class="sm-field sm-field--model">model score $s_\\theta$</div>
        </div>
        <p class="sm-diagram-caption">The quantity we want to minimize — but we
        don't have access to the data score.</p>
      </div>
    </div>

    ${crosslinkSidebar({
      toLesson: 'kl-jensen',
      toAnchor: 'gibbs-inequality',
      toAnchorLabel: 'Non-negativity of KL',
      body: `Fisher divergence is a relative of KL — both measure how well
             $p_\\theta$ approximates $p_{\\mathrm{data}}$, both are non-negative.
             De Bruijn's identity makes the connection precise: Fisher divergence
             is the derivative of KL under Gaussian noise smoothing.`,
    })}
  `;
  container.appendChild(sec);
  renderMath(sec);
}
