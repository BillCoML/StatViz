import { renderMath, mountPrereqStrip } from '@shared/ui';
import { meta } from '../meta';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-1';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="hook-section">
      <div class="hook-coins">
        <div class="coin coin-a">A</div>
        <div class="coin coin-b">B</div>
      </div>
      <h1 class="hook-title">The EM Algorithm</h1>
      <div id="prereq-strip"></div>
      <div class="prose">
        <p>Imagine someone hands you a sequence of coin flip results — say, fifty
        heads-or-tails outcomes. You're told two coins were used, with different
        biases, and that for each block of ten flips one coin was chosen at random.
        But you're not told <em>which</em> coin was used for each block.</p>
        <p>Can you figure out how biased each coin is?</p>
        <p>This is a problem with <strong>missing information</strong> — and it's exactly the kind
        of problem the EM algorithm was built to solve. By the end of this lesson
        you'll know how it works, why it works, and you'll be able to watch it
        work, step by step.</p>
      </div>
      <div class="hook-cta">
        <a href="#section-2" class="cta-btn">Let's set it up →</a>
      </div>
    </div>
  `;
  container.appendChild(sec);
  mountPrereqStrip(sec.querySelector('#prereq-strip') as HTMLElement, {
    prerequisites: meta.prerequisites,
    reasons: {
      'kl-jensen': "You'll need Gibbs' inequality (§5) for the convergence proof in §8.",
    },
  });
  renderMath(sec);
}
