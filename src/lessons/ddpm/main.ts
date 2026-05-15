import '@shared/styles/index.css';
import './styles/overrides.css';

import { mountProgressBar, mountNavSidebar } from '@shared/ui';
import { mount as mountHook }     from './sections/01-hook';
import { mount as mountForward }  from './sections/02-forward-process';
import { mount as mountReverse }  from './sections/03-reverse-process';
import { mount as mountVLB }      from './sections/04-variational-bound';
import { mount as mountPost }     from './sections/05-forward-posterior';
import { mount as mountParam }    from './sections/06-parameterization';
import { mount as mountSMEq }     from './sections/07-score-matching-connection';
import { mount as mountTrain }    from './sections/08-training-algorithm';
import { mount as mountSample }   from './sections/09-sampling-algorithm';
import { mount as mountExplorer } from './sections/10-trained-ddpm-explorer';
import { mount as mountPractical } from './sections/11-practical-considerations';
import { mount as mountWhere }    from './sections/12-where-youll-see-this';

import { mount as mountHero }            from './viz/hero-animation';
import { mount as mountForwardChain }    from './viz/forward-chain-viewer';
import { mount as mountClosedFormJump }  from './viz/closed-form-jump';
import { mount as mountGraphicalModel }  from './viz/graphical-model';
import { mount as mountVLBDecomp }       from './viz/vlb-decomposition';
import { mount as mountFwdPostExplorer } from './viz/forward-posterior-explorer';
import { mount as mountParamCompare }    from './viz/parameterization-comparison';
import { mount as mountScoreEq }         from './viz/score-equivalence';
import { mount as mountTrainTrace }      from './viz/training-trace';
import { mount as mountSamplingChain }   from './viz/sampling-chain';
import { mount as mountTrainedExplorer } from './viz/trained-ddpm-explorer';
import { mount as mountCoarseToFine }    from './viz/coarse-to-fine-interp';
import { mount as mountNoiseSched }      from './viz/noise-schedule-explorer';
import { mount as mountFinalRoadmap }    from './viz/final-roadmap';

const SECTION_LABELS = [
  'Hook',
  'Forward Process',
  'Reverse Process',
  'Variational Bound',
  'Forward Posterior',
  'Parameterization',
  'Score Matching Connection',
  'Training Algorithm',
  'Sampling Algorithm',
  'Trained DDPM Explorer',
  'Practical Considerations',
  "Where You'll See This",
];

const app         = document.getElementById('app')!;
const sidebar     = document.getElementById('sidebar')!;
const hamburger   = document.getElementById('hamburger-menu')!;
const progressBar = document.getElementById('progress-bar')!;

mountHook(app);
mountForward(app);
mountReverse(app);
mountVLB(app);
mountPost(app);
mountParam(app);
mountSMEq(app);
mountTrain(app);
mountSample(app);
mountExplorer(app);
mountPractical(app);
mountWhere(app);

const pick = (id: string) => document.getElementById(id);
if (pick('viz-hero-animation'))            mountHero(pick('viz-hero-animation')!);
if (pick('viz-forward-chain'))             mountForwardChain(pick('viz-forward-chain')!);
if (pick('viz-closed-form-jump'))          mountClosedFormJump(pick('viz-closed-form-jump')!);
if (pick('viz-graphical-model'))           mountGraphicalModel(pick('viz-graphical-model')!);
if (pick('viz-vlb-decomposition'))         mountVLBDecomp(pick('viz-vlb-decomposition')!);
if (pick('viz-forward-posterior-explorer')) mountFwdPostExplorer(pick('viz-forward-posterior-explorer')!);
if (pick('viz-parameterization-comparison')) mountParamCompare(pick('viz-parameterization-comparison')!);
if (pick('viz-score-equivalence'))         mountScoreEq(pick('viz-score-equivalence')!);
if (pick('viz-training-trace'))            mountTrainTrace(pick('viz-training-trace')!);
if (pick('viz-sampling-chain'))            mountSamplingChain(pick('viz-sampling-chain')!);
if (pick('viz-trained-ddpm-explorer'))     mountTrainedExplorer(pick('viz-trained-ddpm-explorer')!);
if (pick('viz-coarse-to-fine-interp'))     mountCoarseToFine(pick('viz-coarse-to-fine-interp')!);
if (pick('viz-noise-schedule-explorer'))   mountNoiseSched(pick('viz-noise-schedule-explorer')!);
if (pick('viz-final-roadmap'))             mountFinalRoadmap(pick('viz-final-roadmap')!);

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
