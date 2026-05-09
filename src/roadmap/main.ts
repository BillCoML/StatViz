import '@shared/styles/index.css';
import './styles/roadmap.css';

import { renderRoadmap } from '@shared/ui/roadmap-graph';
import { renderMobileList } from './mobile-list';
import { mountSidePanel } from './side-panel';

const root = document.getElementById('roadmap-root')!;
const mobileRoot = document.getElementById('roadmap-mobile')!;
const panel = document.getElementById('side-panel')!;
const sidePanel = mountSidePanel(panel);

renderRoadmap(root, {
  mode: 'full',
  onNodeClick: (id) => sidePanel.open(id),
});

renderMobileList(mobileRoot, (id) => sidePanel.open(id));
