# Cross-Page System & Roadmap
## Build Specification for the Lesson Network

> This document defines two things:
>
> 1. The **cross-page system** — the conventions every lesson page
>    follows so they form a coherent network (metadata schema, prereq
>    indicators, cross-link callouts, shared chrome, design tokens).
> 2. The **roadmap page** — the "outside, one step out" interactive map
>    that lets a learner pick a lesson, see its prerequisites, and
>    visualize the path to DDPM.
>
> Read this before building any new lesson page. Existing lessons (the
> EM lesson, completed) need a small retrofit pass to conform — see §5.

---

## 1. Philosophy

The lessons are individually deep; the system that holds them together
should be **almost invisible**. A learner reading the EM lesson should
*feel* it sitting inside a larger structure (via the prereq strip, the
cross-link callouts, the footer roadmap mini-view) but should never feel
the chrome competing for attention with the math.

Three principles:

1. **Lessons are nodes, prerequisites are edges.** The roadmap is a
   directed acyclic graph. Every lesson declares its prereqs in a
   single metadata file; the rest of the system (roadmap rendering,
   prereq strips, crosslinks) is *derived* from that data.

2. **Cross-links are explicit and bidirectional.** When lesson B uses a
   result from lesson A, B has a `<CrosslinkCallout type="back">` at
   the relevant section, and A has a `<CrosslinkCallout type="forward">`
   at the section that produces the result. Both link to the
   counterpart's specific anchor. No dead ends, no silent dependencies.

3. **The system serves the math, not vice versa.** Never sacrifice
   pedagogical depth for "clean architecture." If a sidebar tooltip
   would derail a careful proof, drop the sidebar at that point. The
   chrome is for navigation, not narration.

---

## 2. Lesson Metadata Schema

Every lesson exports a `meta` object from `src/meta.ts`. The roadmap
imports them all and renders.

### Type definition (lives in `@shared/system/types.ts`)

```ts
export type LessonId =
  | 'kl-jensen'
  | 'em'
  | 'elbo-vi'
  | 'vae'
  | 'gaussian-cookbook'
  | 'score-matching'
  | 'ddpm'
  // optional / sidebar lessons:
  | 'normalizing-flows'
  | 'mcmc-foundations'
  | 'metropolis-gibbs'
  | 'langevin';

export type Tier =
  | 1   // foundations (KL, ELBO, Gaussian cookbook)
  | 2   // bridges      (EM, score matching)
  | 3   // applications (VAE, Normalizing flows)
  | 4;  // paper        (DDPM)

export type LessonStatus = 'built' | 'wip' | 'planned';

export interface Prerequisite {
  id: LessonId;
  strength: 'required' | 'recommended';
  // Optional anchor on the prereq page that's specifically used
  // (used by `<PrereqStrip>` to deep-link).
  anchor?: string;
}

export interface LessonMeta {
  id: LessonId;
  title: string;
  subtitle: string;            // one short line, ≤ 80 chars
  tier: Tier;
  difficulty: 1 | 2 | 3 | 4 | 5;
  estimatedHours: number;      // realistic, not breezy
  status: LessonStatus;
  prerequisites: Prerequisite[];
  recommendedNext: LessonId[]; // for "where do I go next" panel
  alsoUsedBy: LessonId[];      // lessons that use results from this one
  description: string;         // 2-3 sentences for the roadmap card
  // Anchors this lesson "exports" — i.e., places other lessons will
  // deep-link to. Maps anchor id → human-readable label.
  exportedAnchors: Record<string, string>;
  // Path on the deployed site (relative root)
  path: string;                // e.g. '/lessons/kl-jensen'
}
```

### The catalog (`@shared/system/catalog.ts`)

```ts
import { meta as klMeta }   from '@lessons/kl-jensen/meta';
import { meta as emMeta }   from '@lessons/em/meta';
import { meta as elboMeta } from '@lessons/elbo-vi/meta';
// ... and so on as lessons are built

export const CATALOG: Record<LessonId, LessonMeta> = {
  'kl-jensen':         klMeta,
  'em':                emMeta,
  'elbo-vi':           elboMeta,
  // ...
};

export const LESSONS_IN_BUILD_ORDER: LessonId[] = [
  'kl-jensen',
  'em',
  'elbo-vi',
  'gaussian-cookbook',
  'vae',
  'score-matching',
  'ddpm',
  // optional sidebars:
  'normalizing-flows',
  'mcmc-foundations',
  'metropolis-gibbs',
  'langevin',
];
```

If a lesson hasn't been built yet, the catalog entry uses a stub `meta`
with `status: 'planned'` and the `path` set to a "coming soon" page.

---

## 3. Cross-Link Conventions

Three flavors of cross-link, each with its own visual treatment.

### 3a. `<PrereqStrip>` — top of every lesson

Renders just below the title on every lesson page. Pulls the
lesson's `prerequisites` array, displays each as a chip with status
badge.

Visual design:

```
┌──────────────────────────────────────────────────────────────────┐
│  Prerequisites:  [✓ KL Divergence & Jensen ▸]  [✓ EM ▸ (recommended)] │
│                  ┌─────────────────────────────────────────────┐ │
│                  │ "You'll need Gibbs' inequality (§5) and the  │ │
│                  │  Gaussian KL formula (§4) from KL & Jensen." │ │
│                  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

Each chip:
- Status icon: ✓ (built), ⏳ (wip), ○ (planned, dimmed)
- Lesson title
- "Recommended" suffix if `strength: 'recommended'`
- Click → navigate to the lesson (with anchor if specified)
- Hover → tooltip with the specific reason this lesson needs this prereq

If the prereq lesson is not yet built (`status: 'planned'`):
- Chip is dimmed
- Hover tooltip: "This prerequisite isn't built yet. You can still
  proceed if you have the background; the relevant background is: …"
  (Pulled from `description` of the planned lesson.)

For lessons with **no** prereqs (foundational), `<PrereqStrip>` shows:
> ✨ **Foundational lesson** — no prerequisites. Recommended starting
> point.

### 3b. `<CrosslinkCallout type="back|forward|sidebar">` — inline in prose

Used at specific points within a section to flag cross-page references.
Three types:

**Type `back`** (looking backward to a result you've already covered):
- Left border in `--ink-soft`
- Icon: ← (left arrow, in `--ink-soft`)
- Header: "**Used by: [target lesson]**"
- Body: short prose explaining the reference
- Link: "Revisit [target lesson §section]"

**Type `forward`** (foreshadowing a result we'll use later):
- Left border in `--amber`
- Icon: → (right arrow, in `--amber`)
- Header: "**Comes back in: [target lesson]**"
- Body: short prose explaining what's coming
- Link: "Continue to [target lesson]" (dimmed if not built)

**Type `sidebar`** (a parallel, non-required reference):
- Left border in `--sage`
- Icon: ↔ (in `--sage`)
- Header: "**Related: [target lesson]**"
- Body: short prose explaining the parallel
- Link: "Explore [target lesson]"

Example markup pattern (TypeScript builders, consumed by the section
modules):

```ts
crosslinkBack({
  toLesson: 'em',
  toAnchor: 'monotonicity',
  body: `The EM convergence proof relied on the inequality
         KL(k(·|x,θ⁽ᵗ⁾) ‖ k(·|x,θ⁽ᵗ⁺¹⁾)) ≥ 0. That's Gibbs' inequality,
         which we just proved.`
})
```

The component renders the link as `${target.path}#${anchor}` and
respects the target's build status (dimmed when planned).

### 3c. `<RoadmapMini>` — embedded in §8 of every lesson

A small (≈300×200px) view of the full roadmap with the current lesson
highlighted as "you are here". Click any node to navigate. Reuses the
roadmap rendering logic — same component, smaller mode.

This sits at the bottom of every lesson and gives the reader an
"oxygen mask" — they can always step back to the map and pick another
direction.

---

## 4. The Roadmap Page

### 4a. Purpose

The single entry point for the lesson network. Visitors land here, see
the structure of the journey from foundations to the DDPM paper, pick
a lesson, and navigate in.

### 4b. URL & file location

- URL: `/` (the site root)
- Source: `roadmap/` package, parallel to lesson packages

```
roadmap/
├── index.html
├── src/
│   ├── main.ts
│   ├── graph-layout.ts        # computes node positions
│   ├── render.ts              # D3-driven SVG render
│   ├── interactions.ts        # hover, click, focus path
│   └── styles/
│       └── roadmap.css
└── package.json
```

### 4c. Information design

The roadmap shows the full lesson graph as a directed acyclic structure.
Three visual encodings carry the information:

1. **Position** encodes tier (left → right):
   - Column 1: Foundations (tier 1)
   - Column 2: Bridges (tier 2)
   - Column 3: Applications (tier 3)
   - Column 4: Paper (tier 4)

2. **Edge style** encodes prereq strength:
   - Solid edge: required prereq
   - Dashed edge: recommended prereq

3. **Node visual** encodes status:
   - Built: full color (lesson's accent or generic ink), normal weight
   - WIP: full color, slight pulse animation
   - Planned: dimmed, dashed border, "coming soon" badge

A **golden thread** (a thicker stroke in `--amber`) traces the
recommended path to DDPM:

```
KL & Jensen ─→ ELBO & VI ─→ VAE ─→ Score Matching ─→ DDPM
                  ↑              ↑
              [Gaussian       [Gaussian
               Cookbook is    Cookbook is
               also used]      also used]
```

EM is shown as a **side excursion** off KL & Jensen — same width edge,
but visually parallel (it's a worthwhile lesson on its own and a
conceptual primer for VAE, but not required for DDPM).

Optional sidebar lessons (Normalizing flows, MCMC, MH/Gibbs, Langevin)
appear in a lower row labeled "Optional side quests" with dimmer
treatment by default.

### 4d. Layout (default state)

ASCII sketch of intended layout — implement with D3 manual positioning,
not force-directed (deterministic placement is more readable for a small
graph):

```
                FOUNDATIONS         BRIDGES          APPLICATIONS         PAPER
    ┌──────────────────────┐   ┌───────────────┐   ┌──────────────┐   ┌─────────┐
    │                      │   │               │   │              │   │         │
    │ ┌────────────────┐   │   │  ┌─────────┐  │   │  ┌────────┐  │   │  ┌────┐ │
    │ │  KL & Jensen ✓ │═══╪═══╪══│ ELBO/VI │══╪═══╪═>│  VAE   │═╪═══╪═>│DDPM│ │
    │ └────────┬───────┘   │   │  └────┬────┘  │   │  └────┬───┘  │   │  └────┘ │
    │          ║           │   │       ║       │   │       ║      │   │     ▲   │
    │          ╚═══════════╪═══╪══>┌───╨───┐   │   │       ║      │   │     ║   │
    │                      │   │   │  EM ✓ │   │   │       ║      │   │     ║   │
    │                      │   │   └───────┘   │   │       ║      │   │     ║   │
    │ ┌────────────────┐   │   │               │   │       ║      │   │     ║   │
    │ │ Gaussian       │═══╪═══╪═══════════════╪═══╪═══════╝      │   │     ║   │
    │ │ Cookbook       │   │   │               │   │              │   │     ║   │
    │ └────────┬───────┘   │   │               │   │              │   │     ║   │
    │          ║           │   │               │   │  ┌────────┐  │   │     ║   │
    │          ╚═══════════╪═══╪═══════════════╪═══╪═>│ Score  │══╪═══╪═════╝   │
    │                      │   │               │   │  │ Match. │  │   │         │
    │                      │   │               │   │  └────────┘  │   │         │
    └──────────────────────┘   └───────────────┘   └──────────────┘   └─────────┘
                                          ┌─────────────────────────────┐
              [Optional side quests]      │ Norm. Flows │ MCMC │ Langevin │
                                          └─────────────────────────────┘
```

Notes on layout:
- Double lines (═══) above are the **golden thread** (recommended path
  to DDPM); single solid lines (───) are required prereqs not on the
  golden thread.
- Edges should be drawn as smooth Bézier curves, not straight or
  orthogonal — softer aesthetic.
- Node spacing: ≥40px vertical between nodes within a column; columns
  spaced so longest edge label fits without overlap.

### 4e. Node design

Each node is a card, ~180px wide × 90px tall:

```
┌─────────────────────────────┐
│  KL & Jensen                │ ← title (Fraunces, weight 600, 16px)
│  Foundations · 2/5 · 3h     │ ← tier · difficulty · time
│                             │
│  The two inequalities that  │ ← subtitle (Source Serif, 13px, italic)
│  hold up everything else.   │
│                          ✓  │ ← status badge (corner)
└─────────────────────────────┘
```

States:
- **Default** (built): paper background, ink text, subtle shadow.
- **Hover**: lifted shadow, ink-bold border in lesson accent color.
- **Focused** (clicked once): persistent highlight, prerequisite
  edges + downstream edges glow (see §4f).
- **Planned**: dashed border, dimmed text, "Coming soon" badge.
- **Current** (when used as `<RoadmapMini>` inside a lesson): bold
  amber border, "You are here" caption.

Status badge icons:
- ✓ (sage) — built, ready to read
- ⏳ (amber) — work in progress
- ○ (ink-soft, dashed) — planned

### 4f. Interactions

1. **Hover a node**: highlight that node + dim all unrelated nodes and
   edges. Show a tooltip with the full description and prereq list.

2. **Click a node**: enter "focus" mode. Prerequisite edges back to the
   roots are highlighted in `--coin-a`. Downstream edges toward DDPM
   are highlighted in `--coin-b`. A side panel slides in from the
   right showing the lesson's full description, anchor list, and a
   prominent **"Open lesson →"** button.

3. **Click an edge**: shows a small popover with "Lesson A uses these
   results from Lesson B: [list of anchors]." (Pulled from B's
   `exportedAnchors` filtered to those referenced in A.)

4. **"Trace the path to DDPM" button** (top-right of the page): plays
   a 3-second sequence highlighting nodes one at a time along the
   golden thread, with a particle following the edges.

5. **Search** (top-left): a small search box. Filter nodes by title.
   Useful when the catalog grows.

6. **Keyboard**: arrow keys move focus between nodes; Enter opens the
   focused lesson; Esc clears focus.

### 4g. Motion design

The roadmap should feel **alive but calm**.

- **On page load**: nodes fade in column-by-column over ~1 second
  (foundations first), then edges draw themselves in a continuous
  stroke animation, then the golden thread highlights with a 600ms
  pulse.
- **On hover**: 150ms ease for highlight transitions.
- **Particle flow** along the golden thread (subtle, optional;
  disabled under `prefers-reduced-motion`): small dots travel from
  KL & Jensen toward DDPM continuously, like a gentle current. Three
  particles at any time, evenly spaced, 6-second cycle. This is the
  "motion where each is built from the foundation of the other"
  the original brief asked for.
- **Trace path animation**: when the user clicks "Trace the path",
  draw a comet-like trail along the golden thread.

All motion respects `prefers-reduced-motion: reduce` — under that
setting, replace fades with instant cuts and disable particles.

### 4h. Side panel

When a node is clicked, a panel (~360px wide) slides in from the right
with three sections:

1. **Header**: lesson title, tier/difficulty/time, status badge.
2. **Description**: the full description from `meta.description`.
3. **Prerequisites**: list of chips (same as `<PrereqStrip>` style).
4. **Used by**: list of `alsoUsedBy` lessons as chips.
5. **Anchors in this lesson**: list of `exportedAnchors` (so a power
   user can see the structure before opening).
6. **Open lesson →** primary button (disabled / "Coming soon" if
   planned).

Esc or click-outside closes the panel.

### 4i. Mobile layout

On viewports < 768px, the horizontal-graph layout breaks down. Switch
to a **vertical stacked list**, grouped by tier:

```
┌────────────────────┐
│  FOUNDATIONS       │
│  ─────────────     │
│  > KL & Jensen ✓   │
│  > Gaussian Cook ○ │
└────────────────────┘
┌────────────────────┐
│  BRIDGES           │
│  ─────────────     │
│  > EM ✓            │
│  > ELBO & VI ○     │
│  > Score Match. ○  │
└────────────────────┘
        ⋮
```

Each item is a tappable row. Tap → opens the side panel as a bottom
sheet.

---

## 5. Retrofit for Existing Lessons (EM)

The EM lesson, already built, predates this system. Updates required:

1. **Add `src/meta.ts`** matching the `LessonMeta` schema. Fill in:
   ```ts
   export const meta: LessonMeta = {
     id: 'em',
     title: 'The EM Algorithm',
     subtitle: 'Finding hidden structure when data is incomplete.',
     tier: 2,
     difficulty: 3,
     estimatedHours: 4,
     status: 'built',
     prerequisites: [
       { id: 'kl-jensen', strength: 'recommended',
         anchor: 'gibbs-inequality' }
     ],
     recommendedNext: ['elbo-vi', 'vae'],
     alsoUsedBy: ['vae'],
     description:
       'The EM algorithm finds maximum-likelihood estimates when there ' +
       'is missing or latent data, by alternating between an E-step ' +
       '(softly imputing the missing data) and an M-step (weighted MLE).',
     exportedAnchors: {
       'monotonicity':         'The monotonic-EM theorem & its proof',
       'q-function':           'The expected complete-data log-likelihood',
       'two-coins-simulator':  'Interactive EM on the two-coins example',
     },
     path: '/lessons/em',
   };
   ```

2. **Add `<PrereqStrip>` at the top of the page**, rendering the
   `kl-jensen` prereq. (Since KL & Jensen is also planned-or-built
   depending on order, the strip will show the appropriate badge.)

3. **Update §8 (the convergence proof)** with a `<CrosslinkCallout
   type="back">` to KL & Jensen §5, replacing the current self-contained
   Gibbs-inequality citation with a proper deep-link.

4. **Add `<RoadmapMini>`** to the bottom of §10 (Summary), highlighting
   EM as "you are here".

5. **Confirm shared chrome** is using the same `tokens.css` as the new
   lessons — extract any duplicated tokens into the shared package.

This is a 1-day retrofit. Schedule it after the KL & Jensen lesson is
built so we have something to link to.

---

## 6. Lesson Catalog (current state)

The catalog as of the writing of this spec:

| ID | Title | Tier | Status | ETA |
|:---|:------|:----:|:------:|:---:|
| `kl-jensen` | KL Divergence & Jensen's Inequality | 1 | planned (next) | 3h |
| `em` | The EM Algorithm | 2 | built | 4h |
| `elbo-vi` | ELBO & Variational Inference | 1 | planned | 3h |
| `gaussian-cookbook` | Gaussian Identities (Cookbook) | 1 | planned | 2h |
| `vae` | Variational Autoencoders | 3 | planned | 3h |
| `score-matching` | Score Matching & Denoising Score Matching | 2 | planned | 3h |
| `ddpm` | Denoising Diffusion Probabilistic Models | 4 | planned | 5h |
| `normalizing-flows` | Normalizing Flows (skim) | 3 | planned | 1h |
| `mcmc-foundations` | Monte Carlo & Markov Chains | 1 | planned (sidebar) | 3h |
| `metropolis-gibbs` | Metropolis–Hastings & Gibbs Sampling | 2 | planned (sidebar) | 3h |
| `langevin` | Langevin Dynamics | 2 | planned (sidebar) | 2h |

The **main path** to DDPM is: **KL & Jensen → ELBO & VI → VAE → Score
Matching → DDPM**, with **Gaussian Cookbook** feeding into ELBO/VI, VAE,
and DDPM. **EM** is a side excursion. **Normalizing flows / MCMC /
MH-Gibbs / Langevin** are optional side quests, useful for breadth or
for follow-up papers (especially score-based SDEs) but not required
for the DDPM paper itself.

---

## 7. Build Sequence

Recommended order to build the system:

1. **Build KL & Jensen** (next). At the same time, extract shared
   chrome (`<NavigationSidebar>`, `<ProgressBar>`, design tokens) from
   the EM lesson into the shared `@shared` package, and create the
   `<PrereqStrip>` and `<CrosslinkCallout>` components.
2. **Retrofit EM** with metadata, `<PrereqStrip>`, the cross-link to KL
   & Jensen, and `<RoadmapMini>` in the footer. (1 day.)
3. **Build the roadmap page** as the new site root, with KL & Jensen
   and EM as the two real (non-planned) nodes.
4. **Build ELBO & VI**. By the time this lesson is built, the system
   has been exercised enough that the spec should be tightened
   wherever rough edges showed up.
5. Continue with VAE → Score Matching → DDPM, with Gaussian Cookbook
   and side quests interleaved as appetite allows.

After step 3, the system is "live" — every new lesson plugs into the
roadmap automatically by exporting `meta`.

---

## 8. Acceptance Criteria

The cross-page system is complete when:

1. Every built lesson exports a valid `meta` object conforming to the
   schema in §2.
2. The roadmap renders the full catalog from `meta` objects alone — no
   hardcoded lesson info in the roadmap source.
3. Clicking any node in the roadmap navigates to the correct lesson
   (or the "coming soon" page for planned ones).
4. Every lesson has a `<PrereqStrip>` at the top reflecting its
   declared prerequisites.
5. Every cross-page reference uses `<CrosslinkCallout>` and renders a
   working deep-link to the target anchor.
6. Every lesson has `<RoadmapMini>` in its summary section,
   highlighting itself.
7. On a fresh visit to the site root, a learner can identify (without
   any prior context) which lesson to start with and which lesson
   leads where.

---

## 9. What This System is NOT

A few non-goals, to keep scope contained:

- **Not a learning management system.** No accounts, no progress
  tracking across visits, no quizzes, no completion certificates.
  Local state (which lessons the user has visited) is OK, but not
  persisted to any backend.
- **Not a wiki.** Lessons are authored as static specs and rendered
  pages. Not user-editable.
- **Not a search engine.** The roadmap has a node-name search but no
  full-text search across lesson content.
- **Not a recommendation engine.** "What should I read next?" is
  answered by the static `recommendedNext` field, not by an algorithm.
- **Not a discussion platform.** No comments, no Q&A, no forum.

If we ever want any of the above, they are separate projects that
**consume** this system, not extensions of it.