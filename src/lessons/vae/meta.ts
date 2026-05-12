import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'vae',
  title: 'Variational Autoencoders',
  subtitle: 'Deep generative models, fit by gradient ascent on the ELBO.',
  tier: 3,
  difficulty: 3,
  estimatedHours: 4,
  status: 'built',
  prerequisites: [
    { id: 'elbo-vi',           strength: 'required',    anchor: 'elbo-two-forms' },
    { id: 'gaussian-cookbook', strength: 'required',    anchor: 'kl-mvn-diag'   },
    { id: 'em',                strength: 'recommended', anchor: 'q-function'    },
  ],
  recommendedNext: ['score-matching', 'ddpm'],
  alsoUsedBy: ['ddpm'],
  description:
    'The variational autoencoder: an amortized variational-inference model with ' +
    'neural-network encoder and decoder, trained end-to-end by gradient ascent on ' +
    'the ELBO. Establishes the architectural pattern that DDPM extends.',
  exportedAnchors: {
    'vae-objective':        'The VAE objective (ELBO with neural encoder/decoder)',
    'amortization':         'Amortized variational inference',
    'reparam-in-vae':       'The reparameterization trick in the VAE pipeline',
    'training-step':        'One full training step, traced numerically',
    'posterior-collapse':   'Posterior collapse: what goes wrong, why, how β-VAE helps',
    'latent-interpolation': 'Latent-space interpolation in a trained VAE',
  },
  path: '/lessons/vae/',
};
