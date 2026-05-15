/**
 * §12: final celebratory roadmap.
 */
import { mountRoadmapMini } from '@shared/ui';

export function mount(container: HTMLElement): void {
  container.innerHTML = `
    <div class="viz-container" style="background:linear-gradient(180deg,var(--paper),#f5e6c5);">
      <div class="viz-title" style="text-align:center;">The curriculum, complete</div>
      <div id="fr-roadmap" style="margin-top:0.6rem;"></div>
      <div class="viz-caption" style="margin-top:1rem;text-align:center;">The golden thread KL → ELBO → VAE & Cookbook → Score Matching → DDPM is fully lit. All seven lessons live.</div>
    </div>`;
  mountRoadmapMini(container.querySelector('#fr-roadmap') as HTMLElement, { currentLessonId: 'ddpm' });
}
