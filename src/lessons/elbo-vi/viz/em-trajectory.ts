import * as d3 from 'd3';
import { eStep, mStep, observedLogLikelihood } from '@lessons/em/em/algorithm';
import { elboTwoCoins } from '../math/em-elbo';

interface TrajectoryPoint {
  label: string;
  logP: number;
  elbo: number;
  klGap: number;
}

function buildTrajectory(): TrajectoryPoint[] {
  const pts: TrajectoryPoint[] = [];

  // Step 0: uninformative q (uniform gammas)
  let thetaA = 0.6, thetaB = 0.5;
  const uniformGammas = [0.5, 0.5, 0.5, 0.5, 0.5];
  let logP0 = observedLogLikelihood(thetaA, thetaB);
  let elbo0 = elboTwoCoins(uniformGammas, thetaA, thetaB);
  pts.push({ label: '0', logP: logP0, elbo: elbo0, klGap: logP0 - elbo0 });

  // E-step at θ⁰
  let resps = eStep(thetaA, thetaB);
  const gammas0 = resps.map(r => r.gammaA);
  let elboAfterE0 = elboTwoCoins(gammas0, thetaA, thetaB);
  pts.push({ label: '0+E', logP: logP0, elbo: elboAfterE0, klGap: logP0 - elboAfterE0 });

  // M-step
  const { thetaA: tA1, thetaB: tB1 } = mStep(resps);
  thetaA = tA1; thetaB = tB1;
  let logP1 = observedLogLikelihood(thetaA, thetaB);
  let elboAfterM0 = elboTwoCoins(gammas0, thetaA, thetaB);
  pts.push({ label: '0+M', logP: logP1, elbo: elboAfterM0, klGap: logP1 - elboAfterM0 });

  // E-step at θ¹
  resps = eStep(thetaA, thetaB);
  const gammas1 = resps.map(r => r.gammaA);
  let elboAfterE1 = elboTwoCoins(gammas1, thetaA, thetaB);
  pts.push({ label: '1+E', logP: logP1, elbo: elboAfterE1, klGap: logP1 - elboAfterE1 });

  // M-step
  const { thetaA: tA2, thetaB: tB2 } = mStep(resps);
  thetaA = tA2; thetaB = tB2;
  let logP2 = observedLogLikelihood(thetaA, thetaB);
  let elboAfterM1 = elboTwoCoins(gammas1, thetaA, thetaB);
  pts.push({ label: '1+M', logP: logP2, elbo: elboAfterM1, klGap: logP2 - elboAfterM1 });

  // Two more iterations
  for (let iter = 2; iter <= 3; iter++) {
    resps = eStep(thetaA, thetaB);
    const gs = resps.map(r => r.gammaA);
    const elboE = elboTwoCoins(gs, thetaA, thetaB);
    const lP = observedLogLikelihood(thetaA, thetaB);
    pts.push({ label: `${iter}+E`, logP: lP, elbo: elboE, klGap: lP - elboE });

    const { thetaA: nA, thetaB: nB } = mStep(resps);
    thetaA = nA; thetaB = nB;
    const lPN = observedLogLikelihood(thetaA, thetaB);
    const elboM = elboTwoCoins(gs, thetaA, thetaB);
    pts.push({ label: `${iter}+M`, logP: lPN, elbo: elboM, klGap: lPN - elboM });
  }

  return pts;
}

export function mountEMTrajectory(container: HTMLElement): void {
  const pts = buildTrajectory();
  const W = 680, H = 280;
  const M = { top: 16, right: 32, bottom: 48, left: 52 };
  const innerW = W - M.left - M.right, innerH = H - M.top - M.bottom;

  container.innerHTML = `
    <div class="em-trajectory">
      <div class="em-trajectory__legend">
        <span>
          <span class="em-trajectory__swatch" style="background:var(--evidence);"></span>
          log p<sub>θ</sub>(x)
        </span>
        <span>
          <span class="em-trajectory__swatch" style="background:var(--elbo);"></span>
          ELBO L(q, θ)
        </span>
      </div>
      <svg id="em-traj-svg" style="width:100%;max-width:${W}px;display:block;"></svg>
      <div style="font-size:0.82em;color:var(--ink-soft);margin-top:0.25rem;">
        E-steps snap the ELBO up to log p(x); M-steps advance both, with ELBO trailing slightly.
      </div>
    </div>
  `;

  const svgEl = container.querySelector('#em-traj-svg') as SVGSVGElement;
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = d3.select(svgEl);
  const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

  const xs = d3.scalePoint<string>().domain(pts.map(p => p.label)).range([0, innerW]).padding(0.2);
  const allVals = [...pts.map(p => p.logP), ...pts.map(p => p.elbo)];
  const ys = d3.scaleLinear().domain([Math.min(...allVals) - 0.3, Math.max(...allVals) + 0.3]).range([innerH, 0]);

  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xs) as any);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(ys).ticks(5) as any);
  g.append('text').attr('class', 'axis-label').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -40).attr('text-anchor', 'middle').text('log-likelihood');

  // Shade the KL gap region between elbo and logP lines
  const gapArea = d3.area<TrajectoryPoint>()
    .x(d => xs(d.label)!)
    .y0(d => ys(d.elbo))
    .y1(d => ys(d.logP))
    .curve(d3.curveLinear);
  g.append('path').attr('fill', 'var(--kl-gap)').attr('fill-opacity', 0.12).attr('d', gapArea(pts)!);

  // log p(x) line
  const logPLine = d3.line<TrajectoryPoint>().x(d => xs(d.label)!).y(d => ys(d.logP));
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--evidence)').attr('stroke-width', 2.5).attr('d', logPLine(pts)!);

  // ELBO line
  const elboLine = d3.line<TrajectoryPoint>().x(d => xs(d.label)!).y(d => ys(d.elbo));
  g.append('path').attr('fill', 'none').attr('stroke', 'var(--elbo)').attr('stroke-width', 2).attr('stroke-dasharray', '5,3').attr('d', elboLine(pts)!);

  // Dots for E-step tightness moments
  g.selectAll('.dot-logp').data(pts).enter().append('circle')
    .attr('cx', d => xs(d.label)!).attr('cy', d => ys(d.logP))
    .attr('r', 3).attr('fill', 'var(--evidence)');
  g.selectAll('.dot-elbo').data(pts).enter().append('circle')
    .attr('cx', d => xs(d.label)!).attr('cy', d => ys(d.elbo))
    .attr('r', 3).attr('fill', 'var(--elbo)');

  // Annotate the first two snap moments (E-step)
  const ePts = pts.filter(p => p.label.includes('+E'));
  g.selectAll('.snap-ann').data(ePts.slice(0, 2)).enter().append('text')
    .attr('x', d => xs(d.label)! + 4).attr('y', d => ys(d.elbo) - 6)
    .attr('font-size', 9).attr('fill', 'var(--elbo)').text('bound tight');
}
