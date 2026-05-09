import type { LessonId } from '../system/types';

export interface RoadmapMiniOptions {
  currentLessonId: LessonId;
}

/**
 * RoadmapMini is the compact view of the lesson graph embedded in lesson summaries.
 * Reuses the full roadmap renderer in mini mode. Imported lazily so the lesson
 * pages don't pull in the full roadmap module at module-load.
 */
export function mountRoadmapMini(container: HTMLElement, opts: RoadmapMiniOptions): void {
  container.innerHTML = `
    <div class="roadmap-mini__title" style="font-family: var(--font-display); font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-soft); margin-bottom: 0.6em;">
      Where this lesson sits
    </div>
    <div class="roadmap-mini__chart"></div>
    <div class="roadmap-mini__footnote" style="font-size: 0.85rem; margin-top: 0.5em;">
      <a href="/">Open the full roadmap →</a>
    </div>
  `;
  const host = container.querySelector('.roadmap-mini__chart') as HTMLElement;

  import('./roadmap-graph').then(({ renderRoadmap }) => {
    renderRoadmap(host, {
      mode: 'mini',
      currentLessonId: opts.currentLessonId,
    });
  });
}
