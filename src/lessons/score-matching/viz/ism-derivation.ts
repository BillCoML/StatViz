import { renderMath } from '@shared/ui';

const STEPS = [
  {
    title: 'Step 1: the score matching objective',
    active: `$$\\mathcal{L}_{\\mathrm{SM}} = \\mathbb{E}_{p_\\mathrm{data}}\\!\\left[\\|s_\\theta(x) - \\nabla \\log p_\\mathrm{data}(x)\\|^2\\right]$$`,
    note: 'We want to minimize the squared distance between the model score and the data score — but we don\'t have access to $\\nabla \\log p_\\mathrm{data}$.',
  },
  {
    title: 'Step 2: expand the squared norm',
    active: `$$= \\mathbb{E}\\!\\left[\\|s_\\theta\\|^2\\right] {\\color{#c87f3b}- 2\\,\\mathbb{E}\\!\\left[s_\\theta^\\top \\nabla \\log p_\\mathrm{data}\\right]} + \\underbrace{\\mathbb{E}\\!\\left[\\|\\nabla \\log p_\\mathrm{data}\\|^2\\right]}_{\\text{const in }\\theta}$$`,
    note: 'The third term is constant in $\\theta$ and can be dropped. The first term is easy. The highlighted middle term is the problem.',
  },
  {
    title: 'Step 3: drop the constant',
    active: `$$\\sim \\mathbb{E}\\!\\left[\\|s_\\theta(x)\\|^2\\right] {\\color{#c87f3b}- 2\\,\\mathbb{E}\\!\\left[s_\\theta(x)^\\top \\nabla \\log p_\\mathrm{data}(x)\\right]}$$`,
    note: 'Now eliminate the highlighted cross term using integration by parts.',
  },
  {
    title: 'Step 4: integration by parts on the cross term',
    active: `$$\\mathbb{E}_{p}\\!\\left[s_{\\theta,i}\\, \\partial_i \\log p\\right] = \\int s_{\\theta,i}\\, \\partial_i p\\, dx = \\underbrace{\\left[s_{\\theta,i}\\, p\\right]_{-\\infty}^{\\infty}}_{0} - \\mathbb{E}_p\\!\\left[\\partial_i s_{\\theta,i}\\right]$$`,
    note: 'The boundary term vanishes (p → 0 at infinity). Summing over all coordinates i: the cross term becomes $-\\mathbb{E}[\\mathrm{tr}(\\nabla s_\\theta)]$.',
  },
  {
    title: 'Step 5: the implicit score matching loss',
    active: `$$\\boxed{\\mathcal{L}_{\\mathrm{ISM}}(\\theta) = \\mathbb{E}_{x \\sim p_\\mathrm{data}}\\!\\left[\\|s_\\theta(x)\\|^2 + 2\\,\\mathrm{tr}\\!\\left(\\nabla s_\\theta(x)\\right)\\right]}$$`,
    note: 'Both terms computable from data samples alone — no $\\nabla \\log p_\\mathrm{data}$ needed. The cost: $\\mathrm{tr}(\\nabla s_\\theta)$ requires $d$ backward passes.',
  },
];

export function mount(container: HTMLElement): void {
  let step = 0;

  container.innerHTML = `
    <div class="viz-container" style="max-width:700px;">
      <div class="viz-title">ISM Derivation</div>
      <div id="ism-step-title" style="font-size:0.95rem;font-weight:600;margin-bottom:0.75rem;color:var(--ink-soft);"></div>
      <div id="ism-step-math" style="overflow-x:auto;min-height:80px;"></div>
      <div id="ism-step-note" style="font-size:0.88rem;color:var(--ink-soft);margin-top:0.75rem;font-style:italic;min-height:48px;"></div>
      <div class="viz-controls" style="margin-top:1rem;">
        <button class="viz-btn" id="ism-prev">← Prev</button>
        <span id="ism-counter" style="font-family:var(--font-mono);font-size:0.85rem;color:var(--ink-soft);"></span>
        <button class="viz-btn" id="ism-next">Next →</button>
        <button class="viz-btn" id="ism-play" style="margin-left:auto;">Play all</button>
      </div>
      <div style="display:flex;gap:4px;margin-top:0.5rem;">
        ${STEPS.map((_, i) => `<div class="ism-dot" data-i="${i}" style="width:12px;height:12px;border-radius:50%;background:#ccc;cursor:pointer;"></div>`).join('')}
      </div>
    </div>
  `;

  const titleEl   = container.querySelector<HTMLElement>('#ism-step-title')!;
  const mathEl    = container.querySelector<HTMLElement>('#ism-step-math')!;
  const noteEl    = container.querySelector<HTMLElement>('#ism-step-note')!;
  const counterEl = container.querySelector<HTMLElement>('#ism-counter')!;
  const prevBtn   = container.querySelector<HTMLButtonElement>('#ism-prev')!;
  const nextBtn   = container.querySelector<HTMLButtonElement>('#ism-next')!;
  const playBtn   = container.querySelector<HTMLButtonElement>('#ism-play')!;
  const dots      = container.querySelectorAll<HTMLElement>('.ism-dot');

  let playTimer: ReturnType<typeof setInterval> | null = null;

  function render() {
    const s = STEPS[step];
    titleEl.textContent = s.title;
    mathEl.innerHTML    = s.active;
    noteEl.innerHTML    = s.note;
    counterEl.textContent = `${step + 1} / ${STEPS.length}`;
    prevBtn.disabled = step === 0;
    nextBtn.disabled = step === STEPS.length - 1;
    dots.forEach((d, i) => {
      d.style.background = i === step ? '#2c5f8d' : i < step ? '#9ab' : '#ccc';
    });
    renderMath(mathEl);
    renderMath(noteEl);
  }

  prevBtn.addEventListener('click', () => { step = Math.max(0, step - 1); render(); });
  nextBtn.addEventListener('click', () => { step = Math.min(STEPS.length - 1, step + 1); render(); });
  dots.forEach(d => d.addEventListener('click', () => { step = parseInt(d.dataset['i']!); render(); }));

  playBtn.addEventListener('click', () => {
    if (playTimer) { clearInterval(playTimer); playTimer = null; playBtn.textContent = 'Play all'; return; }
    step = 0; render();
    playBtn.textContent = 'Stop';
    playTimer = setInterval(() => {
      step++;
      if (step >= STEPS.length) { clearInterval(playTimer!); playTimer = null; playBtn.textContent = 'Play all'; step = STEPS.length - 1; }
      render();
    }, 2200);
  });

  render();
}
