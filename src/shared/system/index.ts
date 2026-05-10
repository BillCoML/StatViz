export type { LessonId, Tier, LessonStatus, Prerequisite, LessonMeta } from './types';
export { CATALOG, LESSONS_IN_BUILD_ORDER, GOLDEN_THREAD, SIDE_QUESTS } from './catalog';

/** Prepends Vite's base URL so links work on GitHub Pages sub-paths. */
export function resolvePath(path: string): string {
  return import.meta.env.BASE_URL + path.slice(1);
}
