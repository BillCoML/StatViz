import '@shared/styles/index.css';
import './styles/overrides.css';

import { mountProgressBar, mountNavSidebar } from '@shared/ui';
import { mount as mountHook }     from './sections/01-hook';
import { mount as mountScore }    from './sections/02-score-function';
import { mount as mountFisher }   from './sections/03-fisher-divergence';
import { mount as mountISM }      from './sections/04-implicit-score-matching';
import { mount as mountDSM }      from './sections/05-denoising-score-matching';
import { mount as mountLangevin } from './sections/06-langevin-dynamics';
import { mount as mountAnnealed } from './sections/07-annealed-langevin';
import { mount as mountWhere }    from './sections/08-where-youll-see-this';

import { mount as mountScoreField } from './viz/score-field-explorer';
import { mount as mountNormIrrel  } from './viz/normalization-irrelevance';
import { mount as mountISMDeriv   } from './viz/ism-derivation';
import { mount as mountNSS        } from './viz/noise-smoothed-score';
import { mount as mountDSMTarget  } from './viz/dsm-target';
import { mount as mountLangSamp   } from './viz/langevin-sampler';
import { mount as mountAnneal     } from './viz/annealed-langevin';

const SECTION_LABELS = [
  'Hook',
  'The Score Function',
  'Fisher Divergence',
  'Implicit Score Matching',
  'Denoising Score Matching',
  'Langevin Dynamics',
  'Annealed Langevin',
  "Where You'll See This",
];

const app         = document.getElementById('app')!;
const sidebar     = document.getElementById('sidebar')!;
const hamburger   = document.getElementById('hamburger-menu')!;
const progressBar = document.getElementById('progress-bar')!;

mountHook(app);
mountScore(app);
mountFisher(app);
mountISM(app);
mountDSM(app);
mountLangevin(app);
mountAnnealed(app);
mountWhere(app);

// Wire visualizations into their placeholder divs
const pick = (id: string) => document.getElementById(id);
if (pick('viz-score-field-explorer')) mountScoreField(pick('viz-score-field-explorer')!);
if (pick('viz-normalization-irrelevance')) mountNormIrrel(pick('viz-normalization-irrelevance')!);
if (pick('viz-ism-derivation'))  mountISMDeriv(pick('viz-ism-derivation')!);
if (pick('viz-noise-smoothed-score')) mountNSS(pick('viz-noise-smoothed-score')!);
if (pick('viz-dsm-target'))      mountDSMTarget(pick('viz-dsm-target')!);
if (pick('viz-langevin-sampler')) mountLangSamp(pick('viz-langevin-sampler')!);
if (pick('viz-annealed-langevin')) mountAnneal(pick('viz-annealed-langevin')!);

mountNavSidebar(sidebar as HTMLElement, hamburger as HTMLElement, { labels: SECTION_LABELS });
mountProgressBar(progressBar as HTMLElement);

document.addEventListener('keydown', (e) => {
  const sections = Array.from(document.querySelectorAll<HTMLElement>('.section'));
  const current = sections.findIndex(s => {
    const rect = s.getBoundingClientRect();
    return rect.top >= -200 && rect.top < window.innerHeight / 2;
  });
  if (e.key === 'ArrowDown' && current < sections.length - 1) {
    sections[current + 1].scrollIntoView({ behavior: 'smooth' });
  } else if (e.key === 'ArrowUp' && current > 0) {
    sections[current - 1].scrollIntoView({ behavior: 'smooth' });
  }
});
