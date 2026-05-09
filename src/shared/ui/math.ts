import 'katex/dist/katex.min.css';
import renderMathInElement from 'katex/contrib/auto-render';

const DELIMITERS = [
  { left: '$$', right: '$$', display: true },
  { left: '$', right: '$', display: false },
];

export function renderMath(el: HTMLElement): void {
  renderMathInElement(el, {
    delimiters: DELIMITERS,
    throwOnError: false,
  });
}
