import '@shared/styles/index.css';
import './styles/overrides.css';

import { mountProgressBar, mountNavSidebar } from '@shared/ui';
import { mount as mountHook }           from './sections/01-hook';
import { mount as mountMVN }            from './sections/02-mvn-foundations';
import { mount as mountKLMV }           from './sections/03-kl-multivariate';
import { mount as mountReparam }        from './sections/04-reparameterization';
import { mount as mountConditioning }   from './sections/05-conditioning';
import { mount as mountLinearGauss }    from './sections/06-linear-gaussian-bayes';
import { mount as mountWhere }          from './sections/07-where-youll-see-this';

const SECTION_LABELS = [
  'Introduction',
  'The Multivariate Gaussian',
  'KL Divergence',
  'Reparameterization',
  'Conditioning',
  'Linear-Gaussian Bayes',
  "Where You'll See This",
];

const app         = document.getElementById('app')!;
const sidebar     = document.getElementById('sidebar')!;
const hamburger   = document.getElementById('hamburger-menu')!;
const progressBar = document.getElementById('progress-bar')!;
const footer      = document.getElementById('footer')!;

mountHook(app);
mountMVN(app);
mountKLMV(app);
mountReparam(app);
mountConditioning(app);
mountLinearGauss(app);
mountWhere(app);

mountNavSidebar(sidebar as HTMLElement, hamburger as HTMLElement, { labels: SECTION_LABELS });
mountProgressBar(progressBar as HTMLElement);

// Update URL hash as user scrolls into sections
const sections = Array.from(document.querySelectorAll<HTMLElement>('.section'));
const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        const id = (entry.target as HTMLElement).dataset.anchor;
        if (id) history.replaceState(null, '', `#${id}`);
      }
    }
  },
  { threshold: 0.3 },
);
sections.forEach(s => { if (s.dataset.anchor) observer.observe(s); });

// Keyboard navigation
document.addEventListener('keydown', (e) => {
  const inView = sections.findIndex(s => {
    const r = s.getBoundingClientRect();
    return r.top <= window.innerHeight * 0.5 && r.bottom >= 0;
  });
  if (e.key === 'ArrowRight' && inView < sections.length - 1) {
    sections[inView + 1].scrollIntoView({ behavior: 'smooth' });
  } else if (e.key === 'ArrowLeft' && inView > 0) {
    sections[inView - 1].scrollIntoView({ behavior: 'smooth' });
  }
});

footer.innerHTML = `<p>A reference for the Gaussian identities that power VAEs, DDPM, and beyond. Built with D3.js and KaTeX.</p>`;

// Scroll to anchor from URL on load
const hash = window.location.hash.slice(1);
if (hash) {
  // Try direct anchor ID first, then data-anchor attribute
  const target = document.getElementById(hash) ||
                 document.querySelector(`[data-anchor="${hash}"]`);
  if (target) setTimeout(() => target.scrollIntoView({ behavior: 'smooth' }), 200);
}
