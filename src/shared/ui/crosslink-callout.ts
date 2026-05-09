import { CATALOG } from '../system/catalog';
import type { LessonId } from '../system/types';

export type CrosslinkType = 'back' | 'forward' | 'sidebar';

export interface CrosslinkCalloutOptions {
  type: CrosslinkType;
  toLesson: LessonId;
  toAnchor?: string;
  /** Optional human-readable section name shown in the link text. */
  toAnchorLabel?: string;
  /** Body prose explaining the cross-reference. */
  body: string;
}

const META: Record<CrosslinkType, { icon: string; headerPrefix: string; linkPrefix: string }> = {
  back:    { icon: '←', headerPrefix: 'Used by',         linkPrefix: 'Revisit'  },
  forward: { icon: '→', headerPrefix: 'Comes back in',   linkPrefix: 'Continue to' },
  sidebar: { icon: '↔', headerPrefix: 'Related',         linkPrefix: 'Explore' },
};

/** Returns the HTML string for a crosslink callout. Use `crosslinkBack/Forward/Sidebar` shortcuts below. */
export function crosslinkCallout(opts: CrosslinkCalloutOptions): string {
  const { type, toLesson, toAnchor, toAnchorLabel, body } = opts;
  const target = CATALOG[toLesson];
  const m = META[type];
  const planned = target.status === 'planned';
  const href = toAnchor ? `${target.path}#${toAnchor}` : target.path;
  const sectionSuffix = toAnchorLabel ? ` §${toAnchorLabel}` : '';
  const linkClass = planned ? 'crosslink__link crosslink__link--dimmed' : 'crosslink__link';
  const linkContent = planned
    ? `${m.linkPrefix} ${target.title} (planned)`
    : `${m.linkPrefix} ${target.title}${sectionSuffix} →`;
  const link = planned
    ? `<span class="${linkClass}" aria-disabled="true">${linkContent}</span>`
    : `<a class="${linkClass}" href="${href}">${linkContent}</a>`;

  const dimClass = planned ? ' crosslink--dimmed' : '';
  return `<aside class="crosslink crosslink--${type}${dimClass}">
    <div class="crosslink__header">
      <span class="crosslink__icon" aria-hidden="true">${m.icon}</span>
      <strong>${m.headerPrefix}: ${target.title}</strong>
    </div>
    <div class="crosslink__body">${body}</div>
    <div class="crosslink__footer">${link}</div>
  </aside>`;
}

export const crosslinkBack = (opts: Omit<CrosslinkCalloutOptions, 'type'>) =>
  crosslinkCallout({ ...opts, type: 'back' });
export const crosslinkForward = (opts: Omit<CrosslinkCalloutOptions, 'type'>) =>
  crosslinkCallout({ ...opts, type: 'forward' });
export const crosslinkSidebar = (opts: Omit<CrosslinkCalloutOptions, 'type'>) =>
  crosslinkCallout({ ...opts, type: 'sidebar' });
