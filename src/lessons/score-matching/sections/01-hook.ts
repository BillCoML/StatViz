import { renderMath, mountPrereqStrip } from '@shared/ui';
import { meta } from '../meta';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-1';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="sm-hook">
      <h1 class="sm-hook__title">Score Matching</h1>
      <p class="sm-hook__subtitle">Model the gradient of the log-density. Sample via Langevin.</p>
      <div class="sm-hook__prose prose">
        <p>Every generative model so far has tried to evaluate or maximize
        $p(x)$, the data density. VAEs gave up on the density directly and
        went after a lower bound. EM iterated on the log-likelihood.
        Plain MLE just sets the gradient to zero.</p>

        <p>But there's another option, and it's a strange one: <strong>don't model
        $p(x)$ at all. Model the gradient $\\nabla_x \\log p(x)$ instead.</strong></p>

        <p>Why would you do that? Two reasons. First, $\\nabla_x \\log p(x)$
        doesn't depend on the normalization constant. If $p(x) = \\tilde{p}(x) / Z$,
        the $\\log Z$ vanishes under the gradient. Models without normalization
        constants are radically more flexible — any function from $\\mathbb{R}^d$
        to $\\mathbb{R}^d$ is a candidate. Second, there's a beautiful sampling
        algorithm — <strong>Langevin dynamics</strong> — that takes a gradient
        field and produces samples from the corresponding distribution.</p>

        <p>The two ideas together — model the score, then sample with Langevin —
        are called <strong>score-based generative modeling</strong>. This lesson
        sets them up. The next lesson, <strong>DDPM</strong>, weaves them into
        the largest family of generative models in the modern world.</p>
      </div>
      <a class="sm-cta" href="#section-2">Meet the score function →</a>
    </div>
    <div id="prereq-strip-container"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountPrereqStrip(
    sec.querySelector('#prereq-strip-container') as HTMLElement,
    { prerequisites: meta.prerequisites },
  );
}
