const SECTION_LABELS = [
  'Hook',
  'The Problem',
  'Complete vs Incomplete',
  'The Key Idea',
  'E-Step',
  'M-Step',
  'Full Algorithm',
  'Why It Works',
  'Pitfalls',
  'Summary',
];

export function mountSidebar(nav: HTMLElement, hamburger: HTMLElement): void {
  // Build hamburger icon
  hamburger.innerHTML = `<button class="hamburger-btn" aria-label="Toggle navigation">
    <span></span><span></span><span></span>
  </button>`;

  // Build TOC links
  const ol = document.createElement('ol');
  ol.className = 'sidebar-toc';

  const links: HTMLAnchorElement[] = [];

  SECTION_LABELS.forEach((label, idx) => {
    const n = idx + 1;
    const li = document.createElement('li');
    li.className = 'sidebar-item';
    const a = document.createElement('a');
    a.href = `#section-${n}`;
    a.className = 'sidebar-link';
    a.textContent = label;
    a.dataset.section = String(n);
    li.appendChild(a);
    ol.appendChild(li);
    links.push(a);
  });

  nav.appendChild(ol);

  // IntersectionObserver to highlight active link
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        const id = entry.target.id; // e.g. "section-3"
        const n = id.replace('section-', '');
        const link = links.find(l => l.dataset.section === n);
        if (link) {
          if (entry.isIntersecting) {
            links.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
          }
        }
      });
    },
    {
      root: null,
      rootMargin: '-20% 0px -60% 0px',
      threshold: 0,
    },
  );

  // Observe sections once they exist (may not exist yet)
  const observeSections = () => {
    for (let n = 1; n <= 10; n++) {
      const el = document.getElementById(`section-${n}`);
      if (el) observer.observe(el);
    }
  };

  // Sections are mounted synchronously before this runs, but just in case:
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', observeSections);
  } else {
    observeSections();
  }

  // Smooth scroll on link click, close sidebar on mobile
  ol.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    if (target.tagName === 'A') {
      e.preventDefault();
      const href = (target as HTMLAnchorElement).getAttribute('href');
      if (href) {
        const el = document.querySelector(href);
        el?.scrollIntoView({ behavior: 'smooth' });
      }
      // Close on mobile
      nav.classList.remove('open');
    }
  });

  // Hamburger toggle
  hamburger.addEventListener('click', () => {
    nav.classList.toggle('open');
  });

  // Close sidebar when clicking outside on mobile
  document.addEventListener('click', (e) => {
    if (!nav.contains(e.target as Node) && !hamburger.contains(e.target as Node)) {
      nav.classList.remove('open');
    }
  });
}
