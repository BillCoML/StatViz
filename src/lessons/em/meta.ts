import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'em',
  title: 'The EM Algorithm',
  subtitle: 'Finding hidden structure when data is incomplete.',
  tier: 2,
  difficulty: 3,
  estimatedHours: 4,
  status: 'built',
  prerequisites: [
    { id: 'kl-jensen', strength: 'recommended', anchor: 'gibbs-inequality' },
  ],
  recommendedNext: ['elbo-vi', 'vae'],
  alsoUsedBy: ['vae'],
  description:
    'The EM algorithm finds maximum-likelihood estimates when there is ' +
    'missing or latent data, by alternating between an E-step (softly ' +
    'imputing the missing data) and an M-step (weighted MLE).',
  exportedAnchors: {
    'monotonicity':        'The monotonic-EM theorem & its proof',
    'q-function':          'The expected complete-data log-likelihood',
    'two-coins-simulator': 'Interactive EM on the two-coins example',
  },
  path: '/lessons/em/',
};
