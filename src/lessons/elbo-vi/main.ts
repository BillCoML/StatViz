import '@shared/styles/index.css';
import './styles/overrides.css';

import { mountProgressBar, mountNavSidebar } from '@shared/ui';
import { mount as mountHook }        from './sections/01-hook';
import { mount as mountSetup }       from './sections/02-the-setup';
import { mount as mountIdentity }    from './sections/03-elbo-identity';
import { mount as mountTwoForms }    from './sections/04-two-forms';
import { mount as mountVIAlgo }      from './sections/05-vi-algorithm';
import { mount as mountWorked }      from './sections/06-worked-examples';
import { mount as mountEMView }      from './sections/07-em-as-elbo-ascent';
import { mount as mountWhere }       from './sections/08-where-youll-see-this';

const SECTION_LABELS = [
  'Hook',
  'The Setup',
  'The ELBO Identity',
  'Two Forms of the ELBO',
  'VI Algorithm',
  'Worked Examples',
  'EM as ELBO Ascent',
  "Where You'll See This",
];

const app        = document.getElementById('app')!;
const sidebar    = document.getElementById('sidebar')!;
const hamburger  = document.getElementById('hamburger-menu')!;
const progressBar = document.getElementById('progress-bar')!;
const footer     = document.getElementById('footer')!;

mountHook(app);
mountSetup(app);
mountIdentity(app);
mountTwoForms(app);
mountVIAlgo(app);
mountWorked(app);
mountEMView(app);
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
