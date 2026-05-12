import * as d3 from 'd3';
import { encoderForward } from '../math/encoder-decoder';
import type { MLPWeights } from '../math/encoder-decoder';

const BETAS         = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
const CLUSTER_COLORS = ['#c0392b','#2c5f8d','#5a8a6a','#d4a437'];

// ─── Shared dataset (matches explorer) ───────────────────────────────────────

function generateDataset(centers: number[][], nPer: number, seed = 42) {
  let s = seed;
  const rng = () => { s=(Math.imul(1664525,s)+1013904223)|0; return (s>>>0)/0x100000000; };
  const rn  = () => { let u; do{u=rng();}while(u===0); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*rng()); };
  const X: [number,number][] = [], labels: number[] = [];
  for(let ci=0;ci<centers.length;ci++)
    for(let i=0;i<nPer;i++){X.push([centers[ci][0]+0.2*rn(),centers[ci][1]+0.2*rn()]);labels.push(ci);}
  return {X,labels};
}

function convertEnc(raw: Record<string,unknown>) {
  return {
    W1: raw['enc1.weight'] as number[][],
    b1: raw['enc1.bias'] as number[],
    W2: [...(raw['enc_mu.weight'] as number[][]), ...(raw['enc_logsigma.weight'] as number[][])],
    b2: [...(raw['enc_mu.bias'] as number[]), ...(raw['enc_logsigma.bias'] as number[])],
  };
}

// ─── Render ───────────────────────────────────────────────────────────────────

export function mountBetaVAESlider(container: HTMLElement): void {
  container.innerHTML = `
    <div class="vae-beta-panel">
      <div>
        <div class="vae-beta-panel__title">Latent space (encoder means)</div>
        <svg id="beta-latent-svg" style="width:100%;aspect-ratio:1;display:block;"></svg>
      </div>
      <div>
        <div class="vae-beta-panel__title">Cluster spread vs β</div>
        <svg id="beta-spread-svg" style="width:100%;aspect-ratio:1;display:block;"></svg>
        <div style="font-size:0.75em;color:var(--ink-soft);margin-top:0.4rem;">
          Cluster spread = mean distance of cluster means from origin.
        </div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.75rem;flex-wrap:wrap;">
      <label class="viz-label" style="display:flex;align-items:center;gap:0.4rem;">
        β
        <input type="range" id="beta-mini-slider" min="0" max="5" step="1" value="2" style="width:140px;">
        <strong id="beta-mini-val" style="font-family:var(--font-mono);min-width:4.5em;">β = 1.0</strong>
      </label>
      <div id="beta-spread-info" style="font-family:var(--font-mono);font-size:0.78em;color:var(--ink-soft);"></div>
    </div>
  `;

  const latentSvgEl = container.querySelector('#beta-latent-svg') as SVGSVGElement;
  const spreadSvgEl = container.querySelector('#beta-spread-svg') as SVGSVGElement;
  const slider      = container.querySelector('#beta-mini-slider') as HTMLInputElement;
  const valEl       = container.querySelector('#beta-mini-val') as HTMLElement;
  const infoEl      = container.querySelector('#beta-spread-info') as HTMLElement;

  const weightsUrl = new URL('../assets/vae-weights.json', import.meta.url).href;

  fetch(weightsUrl)
    .then(r => r.json())
    .then((json: Record<string,unknown>) => {
      const meta = json.metadata as { centers: number[][]; n_per_cluster: number };
      const { X, labels } = generateDataset(meta.centers, meta.n_per_cluster);

      // Pre-compute encoder means for each beta
      const allMus: Record<string,[number,number][]> = {};
      const spreadByBeta: number[] = [];
      for (const beta of BETAS) {
        const raw = json[`beta_${beta}`] as Record<string,unknown>;
        const enc = convertEnc(raw) as MLPWeights;
        const mus = X.map(x => {
          const { mu } = encoderForward(Array.from(x), enc, 2);
          return [mu[0], mu[1]] as [number,number];
        });
        allMus[`b${beta}`] = mus;
        // cluster spread: mean of per-cluster mean norms
        const clusterMeans = meta.centers.map((_, ci) => {
          const pts = mus.filter((_,i) => labels[i] === ci);
          const mx = pts.reduce((s,p)=>s+p[0],0)/pts.length;
          const my = pts.reduce((s,p)=>s+p[1],0)/pts.length;
          return Math.sqrt(mx*mx + my*my);
        });
        spreadByBeta.push(clusterMeans.reduce((s,v)=>s+v,0)/4);
      }

      // Spread line chart
      {
        const W=260,H=180,mL=36,mT=10,mR=10,mB=30;
        spreadSvgEl.setAttribute('viewBox',`0 0 ${W} ${H}`);
        const svg = d3.select(spreadSvgEl);
        const g = svg.append('g').attr('transform',`translate(${mL},${mT})`);
        const iW=W-mL-mR, iH=H-mT-mB;
        const xS = d3.scaleLog().domain([0.25,10]).range([0,iW]);
        const yS = d3.scaleLinear().domain([0,Math.max(...spreadByBeta)*1.2]).range([iH,0]);
        g.append('g').attr('transform',`translate(0,${iH})`).call(d3.axisBottom(xS).ticks(5,'.2~f') as any);
        g.append('g').call(d3.axisLeft(yS).ticks(4) as any);
        g.append('text').attr('transform','rotate(-90)').attr('x',-iH/2).attr('y',-28)
          .attr('text-anchor','middle').attr('font-size',9).attr('fill','var(--ink-soft)').text('cluster spread');
        g.append('text').attr('x',iW/2).attr('y',iH+22)
          .attr('text-anchor','middle').attr('font-size',9).attr('fill','var(--ink-soft)').text('β');
        const line = d3.line<number>().x((_,i)=>xS(BETAS[i])).y(d=>yS(d));
        g.append('path').datum(spreadByBeta).attr('fill','none').attr('stroke','var(--latent-z)').attr('stroke-width',2).attr('d',line(spreadByBeta)!);
        g.selectAll('.dot').data(spreadByBeta).enter().append('circle')
          .attr('cx',(_,i)=>xS(BETAS[i])).attr('cy',d=>yS(d)).attr('r',4)
          .attr('fill','var(--latent-z)').attr('id',(_,i)=>`spread-dot-${i}`);
      }

      let activeBetaIdx = 2;

      function updateLatent(betaIdx: number) {
        const beta = BETAS[betaIdx];
        valEl.textContent = `β = ${beta}`;
        infoEl.textContent = `Cluster spread: ${spreadByBeta[betaIdx].toFixed(3)}`;

        // Highlight dot in spread chart
        d3.select(spreadSvgEl).selectAll('[id^=spread-dot]').attr('r',4).attr('opacity',0.5);
        d3.select(spreadSvgEl).select(`#spread-dot-${betaIdx}`).attr('r',6).attr('opacity',1);

        // Latent scatter
        const mus = allMus[`b${beta}`];
        const W = latentSvgEl.clientWidth || 260;
        const H = latentSvgEl.clientHeight || 260;
        const mL=32,mT=10,mR=10,mB=28;
        latentSvgEl.setAttribute('viewBox',`0 0 ${W} ${H}`);
        const svg = d3.select(latentSvgEl);
        svg.selectAll('*').remove();
        const g = svg.append('g').attr('transform',`translate(${mL},${mT})`);
        const iW=W-mL-mR, iH=H-mT-mB;
        const allX=mus.map(p=>p[0]),allY=mus.map(p=>p[1]);
        const xE=d3.extent(allX) as [number,number], yE=d3.extent(allY) as [number,number];
        const pad=0.3;
        const xS=d3.scaleLinear().domain([Math.min(xE[0],-1)-pad,Math.max(xE[1],1)+pad]).range([0,iW]);
        const yS=d3.scaleLinear().domain([Math.min(yE[0],-1)-pad,Math.max(yE[1],1)+pad]).range([iH,0]);
        g.append('g').attr('transform',`translate(0,${iH})`).call(d3.axisBottom(xS).ticks(4) as any);
        g.append('g').call(d3.axisLeft(yS).ticks(4) as any);
        // prior circle (1σ)
        g.append('circle').attr('cx',xS(0)).attr('cy',yS(0)).attr('r',Math.abs(xS(1)-xS(0)))
          .attr('fill','none').attr('stroke','var(--prior)').attr('stroke-dasharray','4,3').attr('opacity',0.3);
        g.selectAll('.lp').data(mus).enter().append('circle')
          .attr('class','lp').attr('cx',d=>xS(d[0])).attr('cy',d=>yS(d[1]))
          .attr('r',2).attr('fill',(_,i)=>CLUSTER_COLORS[labels[i]]).attr('opacity',0.45);
        // cluster means
        meta.centers.forEach((_,ci)=>{
          const pts=mus.filter((_,i)=>labels[i]===ci);
          const mx=pts.reduce((s,p)=>s+p[0],0)/pts.length;
          const my=pts.reduce((s,p)=>s+p[1],0)/pts.length;
          g.append('circle').attr('cx',xS(mx)).attr('cy',yS(my)).attr('r',5)
            .attr('fill',CLUSTER_COLORS[ci]).attr('opacity',0.9)
            .attr('stroke','var(--paper)').attr('stroke-width',1.5);
        });
      }

      slider.addEventListener('input', () => {
        activeBetaIdx = +slider.value;
        updateLatent(activeBetaIdx);
      });

      updateLatent(activeBetaIdx);
    })
    .catch(() => {
      container.innerHTML += `<p style="color:var(--kl-reg);font-size:0.85em;">
        Could not load vae-weights.json.</p>`;
    });
}
