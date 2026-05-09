export interface NavSidebarOptions {
  labels: string[];
  /** ID prefix for the sections being observed. Sections must have id "{prefix}{n}" with n starting at 1. */
  sectionIdPrefix?: string;
}

export function mountNavSidebar(
  nav: HTMLElement,
  hamburger: HTMLElement,
  opts: NavSidebarOptions,
): void {
  const { labels } = opts;
  const prefix = opts.sectionIdPrefix ?? 'section-';

  hamburger.innerHTML = `<button class="hamburger-btn" aria-label="Toggle navigation">
    <span></span><span></span><span></span>
  </button>`;

  const ol = document.createElement('ol');
  ol.className = 'sidebar-toc';

  const links: HTMLAnchorElement[] = [];

  labels.forEach((label, idx) => {
    const n = idx + 1;
    const li = document.createElement('li');
    li.className = 'sidebar-item';
    const a = document.createElement('a');
    a.href = `#${prefix}${n}`;
    a.className = 'sidebar-link';
    a.textContent = label;
    a.dataset.section = String(n);
    li.appendChild(a);
    ol.appendChild(li);
    links.push(a);
  });

  nav.appendChild(ol);

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        const id = entry.target.id;
        const n = id.replace(prefix, '');
        const link = links.find(l => l.dataset.section === n);
        if (link && entry.isIntersecting) {
          links.forEach(l => l.classList.remove('active'));
          link.classList.add('active');
        }
      });
    },
    { root: null, rootMargin: '-20% 0px -60% 0px', threshold: 0 },
  );

  const observeSections = () => {
    for (let n = 1; n <= labels.length; n++) {
      const el = document.getElementById(`${prefix}${n}`);
      if (el) observer.observe(el);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', observeSections);
  } else {
    observeSections();
  }

  ol.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    if (target.tagName === 'A') {
      e.preventDefault();
      const href = (target as HTMLAnchorElement).getAttribute('href');
      if (href) {
        const el = document.querySelector(href);
        el?.scrollIntoView({ behavior: 'smooth' });
      }
      nav.classList.remove('open');
    }
  });

  hamburger.addEventListener('click', () => {
    nav.classList.toggle('open');
  });

  document.addEventListener('click', (e) => {
    if (!nav.contains(e.target as Node) && !hamburger.contains(e.target as Node)) {
      nav.classList.remove('open');
    }
  });
}
