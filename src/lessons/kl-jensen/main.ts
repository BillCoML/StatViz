import '@shared/styles/index.css';
import './styles/kl-local.css';

import { mountProgressBar, mountNavSidebar } from '@shared/ui';
import { mount as mountHook } from './sections/01-hook';
import { mount as mountConvex } from './sections/02-convex-concave';
import { mount as mountJensen } from './sections/03-jensens-inequality';
import { mount as mountKLDef } from './sections/04-kl-definition';
import { mount as mountNonNeg } from './sections/05-non-negativity';
import { mount as mountProps } from './sections/06-properties';
import { mount as mountForwardReverse } from './sections/07-forward-vs-reverse';
import { mount as mountWhere } from './sections/08-where-youll-see-this';

const SECTION_LABELS = [
  'Hook',
  'Convex & Concave',
  "Jensen's Inequality",
  'KL Divergence',
  'Non-Negativity',
  'Properties',
  'Forward vs Reverse',
  'Where You\'ll See This',
];

const app = document.getElementById('app')!;
const sidebar = document.getElementById('sidebar')!;
const hamburger = document.getElementById('hamburger-menu')!;
const progressBar = document.getElementById('progress-bar')!;
const footer = document.getElementById('footer')!;

mountHook(app);
mountConvex(app);
mountJensen(app);
mountKLDef(app);
mountNonNeg(app);
mountProps(app);
mountForwardReverse(app);
mountWhere(app);

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

footer.innerHTML = `<p>Foundations for the path to DDPM. Built with D3.js and KaTeX.</p>`;
