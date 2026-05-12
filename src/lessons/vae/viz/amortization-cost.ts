export function mountAmortizationCost(container: HTMLElement): void {
  const N = 8;
  let animFrame: number | null = null;
  let perExampleStep = 0;
  let running = false;

  container.innerHTML = `
    <div class="vae-amort">
      <div>
        <div class="vae-amort__panel-title">Per-example VI (no amortization)</div>
        <div id="amort-left" style="display:flex;flex-direction:column;gap:0.3rem;"></div>
        <div class="vae-amort__cost vae-amort__cost--growing" id="amort-left-cost">
          Total cost: <span id="amort-left-n">0</span>× optimization
        </div>
      </div>
      <div>
        <div class="vae-amort__panel-title">Amortized encoder (VAE)</div>
        <div id="amort-right" style="display:flex;flex-direction:column;gap:0.3rem;"></div>
        <div class="vae-amort__cost vae-amort__cost--fixed" id="amort-right-cost">
          Query cost: 1 forward pass (any x)
        </div>
      </div>
    </div>
    <div style="display:flex;gap:0.5rem;margin-top:0.75rem;flex-wrap:wrap;">
      <button class="viz-btn-sm" id="btn-amort-play">▶ Animate</button>
      <button class="viz-btn-sm" id="btn-amort-reset">⟲ Reset</button>
    </div>
  `;

  const leftEl  = container.querySelector('#amort-left') as HTMLElement;
  const rightEl = container.querySelector('#amort-right') as HTMLElement;
  const leftN   = container.querySelector('#amort-left-n') as HTMLElement;

  function makeBar(id: string, frac: number, color: string) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;gap:0.35rem;';
    const label = document.createElement('span');
    label.style.cssText = `font-family:var(--font-mono);font-size:0.7em;min-width:1.8em;color:var(--ink-soft);`;
    label.textContent = `x${id}`;
    const track = document.createElement('div');
    track.style.cssText = 'flex:1;height:10px;background:var(--rule);border-radius:3px;overflow:hidden;';
    const fill = document.createElement('div');
    fill.style.cssText = `height:100%;width:${frac * 100}%;background:${color};border-radius:3px;transition:width 0.3s;`;
    track.appendChild(fill);
    wrap.appendChild(label);
    wrap.appendChild(track);
    return { wrap, fill };
  }

  const leftBars: HTMLElement[] = [];
  const rightBars: HTMLElement[] = [];

  for (let i = 0; i < N; i++) {
    const lb = makeBar(String(i+1), 0, 'var(--kl-reg)');
    leftEl.appendChild(lb.wrap);
    leftBars.push(lb.fill);

    const rb = makeBar(String(i+1), 0, 'var(--encoder)');
    rightEl.appendChild(rb.wrap);
    rightBars.push(rb.fill);
    rb.fill.style.width = '100%';  // encoder is always ready
    rb.fill.style.opacity = '0.4';
  }

  function reset() {
    if (animFrame) cancelAnimationFrame(animFrame);
    animFrame = null;
    running = false;
    perExampleStep = 0;
    leftN.textContent = '0';
    leftBars.forEach(b => { b.style.width = '0%'; });
    rightBars.forEach(b => {
      b.style.width = '100%';
      b.style.opacity = '0.4';
      b.style.background = 'var(--encoder)';
    });
  }

  function animate() {
    if (perExampleStep >= N * 20) { running = false; return; }
    const exIdx = Math.floor(perExampleStep / 20);
    const step = perExampleStep % 20;
    const frac = (step + 1) / 20;
    if (exIdx < N) leftBars[exIdx].style.width = frac * 100 + '%';
    leftN.textContent = String(exIdx + 1);
    perExampleStep++;
    if (running) animFrame = requestAnimationFrame(animate);
  }

  function playRight() {
    rightBars.forEach((b, idx) => {
      setTimeout(() => {
        b.style.opacity = '1';
        b.style.background = 'var(--encoder)';
      }, idx * 40);
    });
  }

  container.querySelector('#btn-amort-play')!.addEventListener('click', () => {
    reset();
    running = true;
    animFrame = requestAnimationFrame(animate);
    playRight();
  });

  container.querySelector('#btn-amort-reset')!.addEventListener('click', reset);

  reset();
}
