import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'elbo-vi',
  title: 'ELBO & Variational Inference',
  subtitle: 'Lower bounds, variational families, and why EM is just coordinate ascent.',
  tier: 1,
  difficulty: 3,
  estimatedHours: 3,
  status: 'built',
  prerequisites: [
    { id: 'kl-jensen', strength: 'required', anchor: 'gibbs-inequality' },
    { id: 'em', strength: 'recommended', anchor: 'q-function' },
  ],
  recommendedNext: ['vae', 'gaussian-cookbook'],
  alsoUsedBy: ['vae', 'ddpm'],
  description:
    'The Evidence Lower Bound (ELBO) is the central objective of modern ' +
    'probabilistic ML. This lesson derives it two ways, shows the ' +
    'fundamental identity log p(x) = ELBO + KL, and revisits EM as ' +
    'coordinate ascent on this surface.',
  exportedAnchors: {
    'fundamental-identity': 'The fundamental identity: log p(x) = ELBO + KL',
    'elbo-two-forms':       'Two equivalent forms of the ELBO',
    'reparam-trick':        'The reparameterization trick (sketch)',
    'em-as-elbo-ascent':    'EM is coordinate ascent on the ELBO',
    'bimodal-failure':      'VI fails on bimodal posteriors with unimodal q',
  },
  path: '/lessons/elbo-vi/',
};
