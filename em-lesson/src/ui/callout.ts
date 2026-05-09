export function callout(type: 'info' | 'tip' | 'warning' | 'proof', title: string, bodyHtml: string): string {
  const icons = { info: 'ℹ', tip: '💡', warning: '⚠', proof: '∴' };
  return `<div class="callout callout-${type}">
    <div class="callout-title">${icons[type]} ${title}</div>
    ${bodyHtml}
  </div>`;
}
