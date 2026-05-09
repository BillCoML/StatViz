export type LessonId =
  | 'kl-jensen'
  | 'em'
  | 'elbo-vi'
  | 'vae'
  | 'gaussian-cookbook'
  | 'score-matching'
  | 'ddpm'
  | 'normalizing-flows'
  | 'mcmc-foundations'
  | 'metropolis-gibbs'
  | 'langevin';

export type Tier = 1 | 2 | 3 | 4;

export type LessonStatus = 'built' | 'wip' | 'planned';

export interface Prerequisite {
  id: LessonId;
  strength: 'required' | 'recommended';
  anchor?: string;
}

export interface LessonMeta {
  id: LessonId;
  title: string;
  subtitle: string;
  tier: Tier;
  difficulty: 1 | 2 | 3 | 4 | 5;
  estimatedHours: number;
  status: LessonStatus;
  prerequisites: Prerequisite[];
  recommendedNext: LessonId[];
  alsoUsedBy: LessonId[];
  description: string;
  exportedAnchors: Record<string, string>;
  path: string;
}
