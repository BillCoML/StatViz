import * as d3 from 'd3';
import { CATALOG, GOLDEN_THREAD } from '@shared/system';
import type { LessonId } from '@shared/system';
import { NODE_POSITIONS, COLUMN_LABELS, SIDE_QUESTS_BAND, DESIGN_BOUNDS } from './roadmap-graph-layout';

export type RoadmapMode = 'full' | 'mini';

export interface RoadmapRenderOptions {
  mode: RoadmapMode;
  /** Highlight this lesson as "you are here". */
  currentLessonId?: LessonId;
  /** Called when user clicks a node (mini: navigates; full: opens side panel). */
  onNodeClick?: (id: LessonId) => void;
}

interface Edge {
  from: LessonId;
  to: LessonId;
  required: boolean;
  onGoldenThread: boolean;
}

function buildEdges(): Edge[] {
  const goldenSet = new Set<string>();
  for (let i = 0; i < GOLDEN_THREAD.length - 1; i++) {
    goldenSet.add(`${GOLDEN_THREAD[i]}->${GOLDEN_THREAD[i + 1]}`);
  }
  const edges: Edge[] = [];
  for (const m of Object.values(CATALOG)) {
    for (const p of m.prerequisites) {
      edges.push({
        from: p.id,
        to: m.id,
        required: p.strength === 'required',
        onGoldenThread: goldenSet.has(`${p.id}->${m.id}`),
      });
    }
  }
  return edges;
}

const STATUS_BADGE: Record<string, string> = { built: '✓', wip: '⏳', planned: '○' };

export function renderRoadmap(host: HTMLElement, opts: RoadmapRenderOptions): void {
  host.innerHTML = '';
  host.classList.add('roadmap', `roadmap--${opts.mode}`);

  const W = DESIGN_BOUNDS.width, H = DESIGN_BOUNDS.height;
  const svg = d3.select(host).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('class', 'roadmap__svg');

  // Column labels (full mode only)
  if (opts.mode === 'full') {
    const labelG = svg.append('g').attr('class', 'roadmap__columns');
    COLUMN_LABELS.forEach(({ x, label }) => {
      labelG.append('text')
        .attr('x', x).attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('class', 'roadmap__col-label')
        .text(label);
    });
    // Side quests band header
    svg.append('text')
      .attr('x', W / 2).attr('y', SIDE_QUESTS_BAND.y - 8)
      .attr('text-anchor', 'middle')
      .attr('class', 'roadmap__band-label')
      .text('Optional side quests');
    svg.append('line')
      .attr('x1', 60).attr('x2', W - 60)
      .attr('y1', SIDE_QUESTS_BAND.y).attr('y2', SIDE_QUESTS_BAND.y)
      .attr('class', 'roadmap__band-rule');
  }

  // Edges
  const edges = buildEdges();
  const positions = new Map(NODE_POSITIONS.map(p => [p.id, p]));
  const NODE_W = 180, NODE_H = 90;
  const NODE_W_MINI = 120, NODE_H_MINI = 60;
  const w = opts.mode === 'mini' ? NODE_W_MINI : NODE_W;
  const h = opts.mode === 'mini' ? NODE_H_MINI : NODE_H;

  const edgeG = svg.append('g').attr('class', 'roadmap__edges');
  for (const e of edges) {
    const a = positions.get(e.from), b = positions.get(e.to);
    if (!a || !b) continue;
    const ax = a.x + w / 2, ay = a.y;
    const bx = b.x - w / 2, by = b.y;
    const cx1 = ax + (bx - ax) * 0.45, cx2 = ax + (bx - ax) * 0.55;
    const path = `M ${ax} ${ay} C ${cx1} ${ay}, ${cx2} ${by}, ${bx} ${by}`;
    const cls = [
      'roadmap__edge',
      e.required ? 'roadmap__edge--required' : 'roadmap__edge--recommended',
      e.onGoldenThread ? 'roadmap__edge--golden' : '',
    ].filter(Boolean).join(' ');
    edgeG.append('path')
      .attr('d', path)
      .attr('class', cls)
      .attr('data-from', e.from).attr('data-to', e.to)
      .attr('fill', 'none');
  }

  // Nodes
  const nodeG = svg.append('g').attr('class', 'roadmap__nodes');
  for (const pos of NODE_POSITIONS) {
    const meta = CATALOG[pos.id];
    const here = pos.id === opts.currentLessonId;
    const groupClass = [
      'roadmap__node',
      `roadmap__node--${meta.status}`,
      pos.isSideQuest ? 'roadmap__node--sidequest' : '',
      here ? 'roadmap__node--here' : '',
    ].filter(Boolean).join(' ');

    const g = nodeG.append('g')
      .attr('class', groupClass)
      .attr('data-id', pos.id)
      .attr('transform', `translate(${pos.x - w / 2}, ${pos.y - h / 2})`);

    // Card rect
    g.append('rect')
      .attr('width', w).attr('height', h)
      .attr('rx', 8).attr('ry', 8)
      .attr('class', 'roadmap__node-card');

    // Title
    g.append('text')
      .attr('class', 'roadmap__node-title')
      .attr('x', 12).attr('y', opts.mode === 'mini' ? 22 : 28)
      .text(opts.mode === 'mini'
        ? truncate(meta.title, 14)
        : truncate(meta.title, 26));

    if (opts.mode === 'full') {
      g.append('text')
        .attr('class', 'roadmap__node-meta')
        .attr('x', 12).attr('y', 46)
        .text(`${tierLabel(meta.tier)} · ${meta.difficulty}/5 · ${meta.estimatedHours}h`);
      g.append('text')
        .attr('class', 'roadmap__node-subtitle')
        .attr('x', 12).attr('y', 68)
        .text(truncate(meta.subtitle, 32));
    }

    // Status badge
    g.append('text')
      .attr('class', 'roadmap__node-badge')
      .attr('x', w - 12).attr('y', opts.mode === 'mini' ? 22 : 24)
      .attr('text-anchor', 'end')
      .text(STATUS_BADGE[meta.status]);

    // "you are here" caption
    if (here && opts.mode === 'mini') {
      g.append('text').attr('class', 'roadmap__node-here')
        .attr('x', w / 2).attr('y', h + 14).attr('text-anchor', 'middle')
        .text('you are here');
    }

    // Click handler
    if (opts.onNodeClick) {
      g.style('cursor', 'pointer')
        .on('click', () => opts.onNodeClick!(pos.id));
    } else if (meta.status !== 'planned') {
      g.style('cursor', 'pointer')
        .on('click', () => { window.location.href = meta.path; });
    }
  }

  // Hover highlight (full mode only)
  if (opts.mode === 'full') {
    nodeG.selectAll<SVGGElement, unknown>('.roadmap__node')
      .on('mouseenter', function() {
        const id = (this as SVGGElement).getAttribute('data-id');
        host.classList.add('roadmap--focused');
        svg.selectAll('.roadmap__node').classed('roadmap__node--related',
          (_d, _i, nodes) => isRelated(id!, (nodes[_i] as SVGGElement).getAttribute('data-id')!));
        svg.selectAll('.roadmap__edge').classed('roadmap__edge--related',
          (_d, _i, nodes) => {
            const path = nodes[_i] as SVGPathElement;
            return path.getAttribute('data-from') === id || path.getAttribute('data-to') === id;
          });
      })
      .on('mouseleave', function() {
        host.classList.remove('roadmap--focused');
        svg.selectAll('.roadmap__node').classed('roadmap__node--related', false);
        svg.selectAll('.roadmap__edge').classed('roadmap__edge--related', false);
      });
  }
}

function tierLabel(t: number): string {
  return t === 1 ? 'Foundations' : t === 2 ? 'Bridges' : t === 3 ? 'Applications' : 'Paper';
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

function isRelated(focused: string, id: string): boolean {
  if (focused === id) return true;
  // Walk prereq chain backward and recommendedNext forward.
  const m = CATALOG[focused as LessonId];
  if (!m) return false;
  if (m.prerequisites.some(p => p.id === id)) return true;
  if (m.recommendedNext.includes(id as LessonId)) return true;
  if (m.alsoUsedBy.includes(id as LessonId)) return true;
  // Reverse: if `id`'s prereqs include `focused`
  const m2 = CATALOG[id as LessonId];
  if (m2 && m2.prerequisites.some(p => p.id === focused)) return true;
  if (m2 && m2.recommendedNext.includes(focused as LessonId)) return true;
  return false;
}
