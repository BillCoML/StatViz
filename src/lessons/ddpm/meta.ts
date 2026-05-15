import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'ddpm',
  title: 'Denoising Diffusion Probabilistic Models',
  subtitle: 'A hierarchical VAE, a score-matching model, and an annealed Langevin sampler — all the same thing.',
  tier: 4,
  difficulty: 5,
  estimatedHours: 6,
  status: 'built',
  prerequisites: [
    { id: 'vae',               strength: 'required',    anchor: 'vae-objective' },
    { id: 'score-matching',    strength: 'required',    anchor: 'dsm' },
    { id: 'gaussian-cookbook', strength: 'required',    anchor: 'conditioning' },
    { id: 'elbo-vi',           strength: 'required',    anchor: 'elbo-two-forms' },
    { id: 'kl-jensen',         strength: 'required',    anchor: 'kl-gaussians' },
    { id: 'em',                strength: 'recommended', anchor: 'q-function' },
  ],
  recommendedNext: [],
  alsoUsedBy: [],
  description:
    'Denoising diffusion probabilistic models: a hierarchical latent ' +
    'variable model with fixed Gaussian forward process and learned ' +
    'reverse Markov chain. Trained as variational inference, with the ' +
    'epsilon-prediction parameterization revealing equivalence to ' +
    'denoising score matching at T noise levels and annealed Langevin ' +
    'sampling. The destination of the StatViz curriculum.',
  exportedAnchors: {
    'forward-process':       'The forward diffusion process (q)',
    'closed-form-marginal':  'The closed-form marginal q(x_t | x_0)',
    'reverse-process':       'The reverse process (p_theta) as a hierarchical VAE',
    'vlb':                   'The variational lower bound and its decomposition',
    'forward-posterior':     'The tractable forward posterior q(x_{t-1} | x_t, x_0)',
    'eps-parameterization':  'The epsilon-prediction parameterization',
    'sm-equivalence':        'The denoising score matching equivalence',
    'training-algorithm':    'Algorithm 1: training',
    'sampling-algorithm':    'Algorithm 2: sampling',
    'L-simple':              'The simplified training objective',
  },
  path: '/lessons/ddpm/',
};
