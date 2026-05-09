import '@shared/styles/index.css';
import './styles/em-local.css';

import { mountProgressBar, mountNavSidebar } from '@shared/ui';
import { mount as mountHook } from './sections/01-hook';
import { mount as mountProblem } from './sections/02-the-problem';
import { mount as mountCompleteVsIncomplete } from './sections/03-complete-vs-incomplete';
import { mount as mountKeyIdea } from './sections/04-the-key-idea';
import { mount as mountEStep } from './sections/05-e-step';
import { mount as mountMStep } from './sections/06-m-step';
import { mount as mountFullAlgorithm } from './sections/07-full-algorithm';
import { mount as mountConvergence } from './sections/08-convergence';
import { mount as mountPitfalls } from './sections/09-pitfalls';
import { mount as mountSummary } from './sections/10-summary';

const SECTION_LABELS = [
  'Hook',
  'The Problem',
  'Complete vs Incomplete',
  'The Key Idea',
  'E-Step',
  'M-Step',
  'Full Algorithm',
  'Why It Works',
  'Pitfalls',
  'Summary',
];

const app = document.getElementById('app')!;
const sidebar = document.getElementById('sidebar')!;
const hamburger = document.getElementById('hamburger-menu')!;
const progressBar = document.getElementById('progress-bar')!;
const footer = document.getElementById('footer')!;

mountHook(app);
mountProblem(app);
mountCompleteVsIncomplete(app);
mountKeyIdea(app);
mountEStep(app);
mountMStep(app);
mountFullAlgorithm(app);
mountConvergence(app);
mountPitfalls(app);
mountSummary(app);

mountNavSidebar(sidebar as HTMLElement, hamburger as HTMLElement, { labels: SECTION_LABELS });
mountProgressBar(progressBar as HTMLElement);

document.addEventListener('keydown', (e) => {
  const sections = Array.from(document.querySelectorAll<HTMLElement>('.section'));
  const inView = sections.findIndex(s => {
    const r = s.getBoundingClientRect();
    return r.top <= window.innerHeight * 0.5 && r.bottom >= 0;
  });
  if (e.key === 'ArrowRight' && inView < sections.length - 1) {
    sections[inView + 1].scrollIntoView({ behavior: 'smooth' });
  } else if (e.key === 'ArrowLeft' && inView > 0) {
    sections[inView - 1].scrollIntoView({ behavior: 'smooth' });
  }
});

footer.innerHTML = `
  <p>Based on Do &amp; Batzoglou, <em>Nature Biotechnology</em> 2008. Built with D3.js and KaTeX.</p>
`;
