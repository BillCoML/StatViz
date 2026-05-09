import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'kl-jensen',
  title: 'KL Divergence & Jensen\'s Inequality',
  subtitle: 'The two inequalities that hold up everything else.',
  tier: 1,
  difficulty: 2,
  estimatedHours: 3,
  status: 'built',
  prerequisites: [],
  recommendedNext: ['elbo-vi'],
  alsoUsedBy: ['em', 'elbo-vi', 'vae', 'ddpm'],
  description:
    'Jensen\'s inequality, KL divergence, the proof that KL ≥ 0, and the ' +
    'forward-vs-reverse-KL distinction that drives variational inference.',
  exportedAnchors: {
    'jensen-statement': 'The statement of Jensen\'s inequality',
    'gibbs-inequality': 'Non-negativity of KL (Gibbs\' inequality)',
    'kl-gaussians':     'Closed form for KL between two Gaussians',
    'reverse-kl':       'Why VI uses reverse KL (mode-seeking)',
  },
  path: '/lessons/kl-jensen/',
};
