import type { LessonId, LessonMeta, Prerequisite } from './types';
import { meta as emMeta } from '@lessons/em/meta';
import { meta as klMeta } from '@lessons/kl-jensen/meta';

const stub = (id: LessonId, title: string, subtitle: string, tier: 1 | 2 | 3 | 4,
              difficulty: 1 | 2 | 3 | 4 | 5, estimatedHours: number,
              description: string, path: string,
              prerequisites: Prerequisite[] = [],
              recommendedNext: LessonId[] = [],
              alsoUsedBy: LessonId[] = []): LessonMeta => ({
  id, title, subtitle, tier, difficulty, estimatedHours,
  status: 'planned',
  prerequisites, recommendedNext, alsoUsedBy,
  description,
  exportedAnchors: {},
  path,
});

const elboStub: LessonMeta = stub(
  'elbo-vi',
  'ELBO & Variational Inference',
  'Turning intractable posteriors into optimization problems.',
  1, 3, 3,
  'The evidence lower bound, the reverse-KL view of variational inference, ' +
  'and how minimizing reverse KL becomes maximizing the ELBO.',
  '/lessons/elbo-vi/',
  [
    { id: 'kl-jensen', strength: 'required' },
    { id: 'gaussian-cookbook', strength: 'recommended' },
  ],
  ['vae'],
  ['vae', 'ddpm'],
);

const gaussianCookbookStub: LessonMeta = stub(
  'gaussian-cookbook',
  'Gaussian Identities (Cookbook)',
  'The Gaussian closures and conditioning rules used everywhere downstream.',
  1, 2, 2,
  'Closed-form identities for Gaussians: marginals, conditionals, products, ' +
  'and KL — the building blocks of every Gaussian-latent variational model.',
  '/lessons/gaussian-cookbook/',
  [],
  ['elbo-vi', 'vae'],
  ['elbo-vi', 'vae', 'score-matching', 'ddpm'],
);

const vaeStub: LessonMeta = stub(
  'vae',
  'Variational Autoencoders',
  'A neural network amortizes the variational posterior.',
  3, 3, 3,
  'The VAE objective: reconstruction + KL regularizer. The reparameterization ' +
  'trick. Why the encoder is amortized inference.',
  '/lessons/vae/',
  [
    { id: 'elbo-vi', strength: 'required' },
    { id: 'gaussian-cookbook', strength: 'required' },
    { id: 'em', strength: 'recommended' },
  ],
  ['score-matching', 'ddpm'],
  ['ddpm'],
);

const scoreMatchingStub: LessonMeta = stub(
  'score-matching',
  'Score Matching & Denoising Score Matching',
  'Learn the gradient of the log density without ever computing the density.',
  2, 4, 3,
  'Score matching, denoising score matching, and the connection to noise-' +
  'conditional density estimation that DDPM exploits.',
  '/lessons/score-matching/',
  [
    { id: 'kl-jensen', strength: 'required' },
    { id: 'gaussian-cookbook', strength: 'recommended' },
  ],
  ['ddpm'],
  ['ddpm', 'langevin'],
);

const ddpmStub: LessonMeta = stub(
  'ddpm',
  'Denoising Diffusion Probabilistic Models',
  'The destination paper.',
  4, 5, 5,
  'The Ho/Jain/Abbeel 2020 paper read in full. The forward noising chain, the ' +
  'reverse denoising chain, the simplified training objective, sampling.',
  '/lessons/ddpm/',
  [
    { id: 'vae', strength: 'required' },
    { id: 'score-matching', strength: 'required' },
    { id: 'gaussian-cookbook', strength: 'required' },
  ],
);

const normalizingFlowsStub: LessonMeta = stub(
  'normalizing-flows',
  'Normalizing Flows (skim)',
  'Invertible neural networks as exact-likelihood density estimators.',
  3, 3, 1,
  'A short tour: change-of-variables, coupling layers, and where flows sit ' +
  'next to VAEs and diffusion.',
  '/lessons/normalizing-flows/',
  [{ id: 'vae', strength: 'recommended' }],
);

const mcmcStub: LessonMeta = stub(
  'mcmc-foundations',
  'Monte Carlo & Markov Chains',
  'The classical sampling toolkit, side-quest edition.',
  1, 2, 3,
  'Monte Carlo estimation, Markov chains, stationary distributions, ergodicity ' +
  '— the foundations under MH and Gibbs sampling.',
  '/lessons/mcmc-foundations/',
  [],
  ['metropolis-gibbs', 'langevin'],
  ['metropolis-gibbs', 'langevin'],
);

const mhGibbsStub: LessonMeta = stub(
  'metropolis-gibbs',
  'Metropolis–Hastings & Gibbs Sampling',
  'The classical posterior-sampling algorithms.',
  2, 3, 3,
  'Metropolis–Hastings as a general-purpose MCMC sampler; Gibbs sampling as ' +
  'its conditional special case.',
  '/lessons/metropolis-gibbs/',
  [{ id: 'mcmc-foundations', strength: 'required' }],
);

const langevinStub: LessonMeta = stub(
  'langevin',
  'Langevin Dynamics',
  'Sampling via gradient flow plus noise.',
  2, 4, 2,
  'Langevin dynamics as gradient-based sampling, and its connection to ' +
  'score-based generative models.',
  '/lessons/langevin/',
  [
    { id: 'mcmc-foundations', strength: 'required' },
    { id: 'score-matching', strength: 'recommended' },
  ],
);

export const CATALOG: Record<LessonId, LessonMeta> = {
  'kl-jensen':         klMeta,
  'em':                emMeta,
  'elbo-vi':           elboStub,
  'gaussian-cookbook': gaussianCookbookStub,
  'vae':               vaeStub,
  'score-matching':    scoreMatchingStub,
  'ddpm':              ddpmStub,
  'normalizing-flows': normalizingFlowsStub,
  'mcmc-foundations':  mcmcStub,
  'metropolis-gibbs':  mhGibbsStub,
  'langevin':          langevinStub,
};

export const LESSONS_IN_BUILD_ORDER: LessonId[] = [
  'kl-jensen',
  'em',
  'elbo-vi',
  'gaussian-cookbook',
  'vae',
  'score-matching',
  'ddpm',
  'normalizing-flows',
  'mcmc-foundations',
  'metropolis-gibbs',
  'langevin',
];

export const GOLDEN_THREAD: LessonId[] = [
  'kl-jensen',
  'elbo-vi',
  'vae',
  'score-matching',
  'ddpm',
];

/** Lessons treated as side quests in the roadmap (lower row, dimmer treatment). */
export const SIDE_QUESTS: LessonId[] = [
  'normalizing-flows',
  'mcmc-foundations',
  'metropolis-gibbs',
  'langevin',
];
