import { CATALOG, LESSONS_IN_BUILD_ORDER } from '@shared/system';
import type { LessonId, Tier } from '@shared/system';

const STATUS_BADGE: Record<string, string> = { built: '✓', wip: '⏳', planned: '○' };
const TIER_LABEL: Record<Tier, string> = { 1: 'Foundations', 2: 'Bridges', 3: 'Applications', 4: 'Paper' };

export function renderMobileList(host: HTMLElement, onClick: (id: LessonId) => void): void {
  const groups = new Map<Tier, LessonId[]>();
  for (const id of LESSONS_IN_BUILD_ORDER) {
    const tier = CATALOG[id].tier;
    if (!groups.has(tier)) groups.set(tier, []);
    groups.get(tier)!.push(id);
  }

  host.classList.add('roadmap-mobile');
  host.innerHTML = '';
  for (const tier of [1, 2, 3, 4] as Tier[]) {
    const ids = groups.get(tier);
    if (!ids) continue;
    const groupEl = document.createElement('div');
    groupEl.className = 'roadmap-mobile__group';
    groupEl.innerHTML = `<div class="roadmap-mobile__group-title">${TIER_LABEL[tier]}</div>`;
    for (const id of ids) {
      const m = CATALOG[id];
      const a = document.createElement('a');
      a.className = 'roadmap-mobile__row' + (m.status === 'planned' ? ' roadmap-mobile__row--planned' : '');
      a.href = '#';
      a.innerHTML = `
        <span><strong>${m.title}</strong></span>
        <span style="font-family: var(--font-mono); color: var(--ink-soft);">
          ${STATUS_BADGE[m.status]} · ${m.estimatedHours}h
        </span>
      `;
      a.addEventListener('click', (e) => { e.preventDefault(); onClick(id); });
      groupEl.appendChild(a);
    }
    host.appendChild(groupEl);
  }
}
