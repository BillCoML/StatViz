import type { LessonId } from '@shared/system';

export interface NodePosition {
  id: LessonId;
  x: number;
  y: number;
  isSideQuest: boolean;
}

/**
 * Hand-positioned coordinates for the lesson graph. Coordinates are in an
 * abstract "design space"; the renderer scales them to fit the viewport.
 */
export const NODE_POSITIONS: NodePosition[] = [
  // Foundations column (tier 1, x=100)
  { id: 'kl-jensen',         x: 100, y: 100, isSideQuest: false },
  { id: 'gaussian-cookbook', x: 100, y: 260, isSideQuest: false },
  // Bridges column (tier 2, x=380)
  { id: 'elbo-vi',           x: 380, y: 100, isSideQuest: false },
  { id: 'em',                x: 380, y: 260, isSideQuest: false },
  { id: 'score-matching',    x: 380, y: 400, isSideQuest: false },
  // Applications column (tier 3, x=660)
  { id: 'vae',               x: 660, y: 100, isSideQuest: false },
  // Paper column (tier 4, x=940)
  { id: 'ddpm',              x: 940, y: 200, isSideQuest: false },
  // Side quests row (y=560)
  { id: 'mcmc-foundations',  x: 100, y: 560, isSideQuest: true },
  { id: 'metropolis-gibbs',  x: 380, y: 560, isSideQuest: true },
  { id: 'langevin',          x: 380, y: 680, isSideQuest: true },
  { id: 'normalizing-flows', x: 660, y: 560, isSideQuest: true },
];

export const COLUMN_LABELS = [
  { x: 100, label: 'Foundations' },
  { x: 380, label: 'Bridges' },
  { x: 660, label: 'Applications' },
  { x: 940, label: 'Paper' },
];

export const SIDE_QUESTS_BAND = { y: 510, height: 250 };

/** The dimensions of the design coordinate space. */
export const DESIGN_BOUNDS = { width: 1080, height: 800 };
