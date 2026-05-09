export type CalloutType = 'info' | 'tip' | 'warning' | 'proof';

const ICONS: Record<CalloutType, string> = {
  info: 'ℹ',
  tip: '💡',
  warning: '⚠',
  proof: '∴',
};

export function callout(type: CalloutType, title: string, bodyHtml: string): string {
  return `<div class="callout callout-${type}">
    <div class="callout-title">${ICONS[type]} ${title}</div>
    ${bodyHtml}
  </div>`;
}
