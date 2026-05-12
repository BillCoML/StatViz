import '@shared/styles/index.css';
import './styles/overrides.css';

import { mountProgressBar, mountNavSidebar } from '@shared/ui';
import { mount as mountHook }     from './sections/01-hook';
import { mount as mountSetup }    from './sections/02-the-setup';
import { mount as mountAmort }    from './sections/03-amortized-inference';
import { mount as mountObj }      from './sections/04-the-vae-objective';
import { mount as mountKL }       from './sections/05-closed-form-kl';
import { mount as mountReparam }  from './sections/06-reparameterization';
import { mount as mountTrace }    from './sections/07-one-training-step';
import { mount as mountExplorer } from './sections/08-trained-vae-explorer';
import { mount as mountFailures } from './sections/09-failure-modes';
import { mount as mountWhere }    from './sections/10-where-youll-see-this';

const SECTION_LABELS = [
  'Hook',
  'The Setup',
  'Amortized Inference',
  'The VAE Objective',
  'Closed-Form KL',
  'Reparameterization',
  'One Training Step',
  'Trained VAE Explorer',
  'Failure Modes',
  "Where You'll See This",
];

const app         = document.getElementById('app')!;
const sidebar     = document.getElementById('sidebar')!;
const hamburger   = document.getElementById('hamburger-menu')!;
const progressBar = document.getElementById('progress-bar')!;
const footer      = document.getElementById('footer')!;

mountHook(app);
mountSetup(app);
mountAmort(app);
mountObj(app);
mountKL(app);
mountReparam(app);
mountTrace(app);
mountExplorer(app);
mountFailures(app);
mountWhere(app);

mountNavSidebar(sidebar as HTMLElement, hamburger as HTMLElement, { labels: SECTION_LABELS });
mountProgressBar(progressBar as HTMLElement);

document.addEventListener('keydown', (e) => {
  const sections = Array.from(document.querySelectorAll<HTMLElement>('.section'));
  const inView = sections.findIndex(s => {
    const r = s.getBoundingClientRect();
    return r.top <= window.innerHeight * 0.5 && r.bottom >= 0;
  });
  if (e.key === 'ArrowRight' && inView < sections.length - 1)
    sections[inView + 1].scrollIntoView({ behavior: 'smooth' });
  else if (e.key === 'ArrowLeft' && inView > 0)
    sections[inView - 1].scrollIntoView({ behavior: 'smooth' });
});

footer.innerHTML = `<p>Foundations for the path to DDPM. Built with D3.js and KaTeX.</p>`;
