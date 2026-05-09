declare global {
  interface Window {
    renderMathInElement: (el: HTMLElement, opts: object) => void;
    __katexLoaded?: boolean;
    __katexReadyCallback?: () => void;
  }
}

export function renderMath(el: HTMLElement): void {
  const opts = {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
  };
  if (window.__katexLoaded && window.renderMathInElement) {
    window.renderMathInElement(el, opts);
  } else {
    window.__katexReadyCallback = () => {
      window.renderMathInElement(el, opts);
    };
  }
}
