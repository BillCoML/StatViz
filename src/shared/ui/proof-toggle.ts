export function proofToggle(title: string, bodyHtml: string, openByDefault = false): string {
  return `<details class="proof-toggle" ${openByDefault ? 'open' : ''}>
    <summary>${title}</summary>
    <div class="proof-body">${bodyHtml}</div>
  </details>`;
}
