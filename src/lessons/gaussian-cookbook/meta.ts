import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'gaussian-cookbook',
  title: 'Gaussian Cookbook',
  subtitle: "The four Gaussian identities you'll keep using.",
  tier: 1,
  difficulty: 2,
  estimatedHours: 2,
  status: 'built',
  prerequisites: [
    {
      id: 'kl-jensen',
      strength: 'required',
      anchor: 'kl-gaussians',
    },
  ],
  recommendedNext: ['vae', 'ddpm'],
  alsoUsedBy: ['vae', 'ddpm', 'score-matching'],
  description:
    'Four Gaussian identities collected as a reference: multivariate ' +
    'KL, the reparameterization trick in matrix form, conditioning ' +
    'and marginalization of jointly Gaussian variables, and the ' +
    'linear-Gaussian Bayesian update.',
  exportedAnchors: {
    'mvn-density':        'The multivariate Gaussian density',
    'kl-mvn':             'KL divergence between multivariate Gaussians',
    'kl-mvn-diag':        'KL when both covariances are diagonal (VAE regularizer)',
    'reparam-matrix':     'The reparameterization trick (matrix form)',
    'conditioning':       'Conditional & marginal of jointly Gaussian variables',
    'linear-gauss-bayes': 'Linear-Gaussian Bayesian update (closed-form posterior)',
  },
  path: '/lessons/gaussian-cookbook/',
};
