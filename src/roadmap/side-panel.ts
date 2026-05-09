import { CATALOG } from '@shared/system';
import type { LessonId } from '@shared/system';

const STATUS_BADGE: Record<string, string> = { built: '✓', wip: '⏳', planned: '○' };

export function mountSidePanel(host: HTMLElement) {
  let openId: LessonId | null = null;

  function close() {
    openId = null;
    host.classList.remove('side-panel--open');
    host.innerHTML = '';
  }

  function open(id: LessonId) {
    openId = id;
    const m = CATALOG[id];
    const planned = m.status === 'planned';
    const prereqChips = m.prerequisites.length
      ? m.prerequisites.map(p => {
          const t = CATALOG[p.id];
          return `<a class="side-panel__chip" href="${t.path}">${STATUS_BADGE[t.status]} ${t.title}${p.strength === 'recommended' ? ' (recommended)' : ''}</a>`;
        }).join('')
      : '<span class="side-panel__none">Foundational — no prerequisites.</span>';
    const usedByChips = m.alsoUsedBy.length
      ? m.alsoUsedBy.map(id => {
          const t = CATALOG[id];
          return `<a class="side-panel__chip" href="${t.path}">${STATUS_BADGE[t.status]} ${t.title}</a>`;
        }).join('')
      : '<span class="side-panel__none">Not yet used downstream.</span>';
    const anchorList = Object.entries(m.exportedAnchors).length
      ? Object.entries(m.exportedAnchors).map(([anc, label]) =>
          `<li><a href="${m.path}#${anc}">${label}</a></li>`).join('')
      : '<li class="side-panel__none">No exported anchors yet.</li>';

    host.classList.add('side-panel--open');
    host.innerHTML = `
      <div class="side-panel__inner">
        <button class="side-panel__close" aria-label="Close">×</button>
        <header class="side-panel__header">
          <h2 class="side-panel__title">${m.title}</h2>
          <div class="side-panel__meta">
            ${tierLabel(m.tier)} · difficulty ${m.difficulty}/5 · ${m.estimatedHours}h ·
            <span class="side-panel__status side-panel__status--${m.status}">${STATUS_BADGE[m.status]} ${m.status}</span>
          </div>
        </header>
        <p class="side-panel__desc">${m.description}</p>
        <h3>Prerequisites</h3>
        <div class="side-panel__chips">${prereqChips}</div>
        <h3>Used by</h3>
        <div class="side-panel__chips">${usedByChips}</div>
        <h3>Anchors in this lesson</h3>
        <ul class="side-panel__anchors">${anchorList}</ul>
        <div class="side-panel__actions">
          ${planned
            ? `<span class="cta-btn" style="opacity: 0.5; cursor: not-allowed;">Coming soon</span>`
            : `<a class="cta-btn" href="${m.path}">Open lesson →</a>`}
        </div>
      </div>
    `;
    host.querySelector('.side-panel__close')!.addEventListener('click', close);
  }

  // Click-outside / Esc
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
  host.addEventListener('click', e => { if (e.target === host) close(); });

  return {
    open,
    close,
    isOpen: () => openId !== null,
  };
}

function tierLabel(t: number): string {
  return t === 1 ? 'Foundations' : t === 2 ? 'Bridges' : t === 3 ? 'Applications' : 'Paper';
}
