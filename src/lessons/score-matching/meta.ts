import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'score-matching',
  title: 'Score Matching',
  subtitle: 'Model the gradient of the log-density. Sample via Langevin.',
  tier: 3,
  difficulty: 3,
  estimatedHours: 4,
  status: 'built',
  prerequisites: [
    { id: 'kl-jensen',         strength: 'required',    anchor: 'gibbs-inequality' },
    { id: 'gaussian-cookbook', strength: 'required',    anchor: 'reparam-matrix'   },
  ],
  recommendedNext: ['ddpm'],
  alsoUsedBy: ['ddpm'],
  description:
    'Score-based generative modeling: fit ∇ log p(x) directly, ' +
    'avoiding the normalization constant. Three tractable losses, ' +
    'Langevin sampling, and the noise-annealing trick that DDPM ' +
    'extends into a Markov chain.',
  exportedAnchors: {
    'score-definition':  'The score function: ∇_x log p(x)',
    'fisher-divergence': 'Fisher divergence and the score matching objective',
    'ism':               'Implicit score matching (Hyvärinen integration by parts)',
    'dsm':               'Denoising score matching (Vincent identity)',
    'langevin':          'Langevin dynamics for sampling from a known score',
    'annealed-langevin': 'Annealed Langevin sampling across noise levels',
  },
  path: '/lessons/score-matching/',
};
