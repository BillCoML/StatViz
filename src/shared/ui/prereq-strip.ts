import { CATALOG } from '../system/catalog';
import { resolvePath } from '../system';
import type { LessonId, LessonStatus, Prerequisite } from '../system/types';

export interface PrereqStripOptions {
  prerequisites: Prerequisite[];
  /** Optional per-prereq tooltip explaining the specific dependency. Keyed by prereq id. */
  reasons?: Partial<Record<LessonId, string>>;
}

const STATUS_ICON: Record<LessonStatus, string> = {
  built:   '✓',
  wip:     '⏳',
  planned: '○',
};

export function mountPrereqStrip(container: HTMLElement, opts: PrereqStripOptions): void {
  const { prerequisites, reasons = {} } = opts;

  if (prerequisites.length === 0) {
    container.innerHTML = `<div class="prereq-strip prereq-strip--empty">
      <span class="prereq-strip__sparkle">✨</span>
      <strong>Foundational lesson</strong> — no prerequisites. Recommended starting point.
    </div>`;
    return;
  }

  const chips = prerequisites.map(p => {
    const target = CATALOG[p.id];
    const icon = STATUS_ICON[target.status];
    const recommended = p.strength === 'recommended' ? ' <span class="prereq-strip__suffix">(recommended)</span>' : '';
    const dimmed = target.status === 'planned' ? ' prereq-strip__chip--dimmed' : '';
    const href = p.anchor ? `${resolvePath(target.path)}#${p.anchor}` : resolvePath(target.path);
    const reason = reasons[p.id]
      ?? (target.status === 'planned'
            ? `This prerequisite isn't built yet. Background: ${target.description}`
            : '');
    const tooltip = reason ? ` data-tooltip="${escapeAttr(reason)}"` : '';
    return `<a class="prereq-strip__chip${dimmed}" href="${href}"${tooltip}>
      <span class="prereq-strip__icon">${icon}</span>
      <span class="prereq-strip__title">${target.title}</span>${recommended}
    </a>`;
  }).join('');

  container.innerHTML = `<div class="prereq-strip">
    <span class="prereq-strip__label">Prerequisites:</span>
    <div class="prereq-strip__chips">${chips}</div>
  </div>`;
}

function escapeAttr(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
