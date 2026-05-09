export function mountProgressBar(el: HTMLElement): void {
  const update = () => {
    const h = document.documentElement.scrollHeight - window.innerHeight;
    el.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + '%';
  };
  window.addEventListener('scroll', update, { passive: true });
  update();
}
