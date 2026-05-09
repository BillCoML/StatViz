# KL Divergence & Jensen's Inequality — Interactive Lesson
## Build Specification & Content Plan

> A single-page, deeply explained, interactive lesson on Jensen's inequality
> and the Kullback–Leibler divergence. These two ideas are the structural
> backbone of variational inference, EM, VAEs, and DDPM. This page earns the
> reader the right to use them.
>
> **Position in the roadmap**: foundational. Required for ELBO, VAE, DDPM.
> Used (without proof) in the EM convergence theorem; this page back-fills
> that proof.

---

## 0. Pedagogical Philosophy

Same commitments as the EM lesson:

1. **Concrete before abstract.** Every theorem gets a worked numerical
   example *before* the general statement is proved.
2. **Math is shown in full.** Both the convex-combination proof of Jensen
   and the supporting-hyperplane proof. The Gibbs proof. The Gaussian-KL
   derivation in full algebra.
3. **Visualization carries half the load.** Every abstract object — the
   chord-above-the-curve, the asymmetry of KL, the mode-seeking vs
   mean-seeking dichotomy — gets a manipulable visualization.
4. **Cross-page integration is explicit.** This is the first lesson built
   under the cross-page system (see `SYSTEM_AND_ROADMAP.md`). The
   non-negativity proof links *back* to the EM convergence theorem (which
   used it as a black box). The forward/reverse-KL section links *forward*
   to the upcoming ELBO lesson.

By the end, the reader can: (a) state and prove Jensen's for both
discrete and continuous random variables, (b) write the KL definition
from memory and recognize when it's infinite, (c) prove the
non-negativity of KL, (d) derive the closed-form KL between two
Gaussians, and (e) explain why variational inference uses *reverse* KL
and what behavior that produces.

---

## 1. Tech Stack

Identical to EM lesson. Vite + TypeScript, KaTeX (CDN auto-render), D3 v7,
hand-written CSS with the shared token system, Fraunces / Source Serif 4 /
JetBrains Mono.

File layout (mirrors EM):

```
kl-lesson/
├── index.html
├── src/
│   ├── main.ts
│   ├── katex-render.ts
│   ├── meta.ts                      # lesson metadata (per system spec)
│   ├── math/
│   │   ├── kl.ts                    # KL between common distributions
│   │   ├── jensen.ts                # numeric demos of Jensen's gap
│   │   └── kl.test.ts
│   ├── sections/
│   │   ├── 01-hook.ts
│   │   ├── 02-convex-concave.ts
│   │   ├── 03-jensens-inequality.ts
│   │   ├── 04-kl-definition.ts
│   │   ├── 05-non-negativity.ts
│   │   ├── 06-properties.ts
│   │   ├── 07-forward-vs-reverse.ts
│   │   └── 08-where-youll-see-this.ts
│   ├── viz/
│   │   ├── convex-explorer.ts       # §2
│   │   ├── jensen-gap.ts            # §3
│   │   ├── kl-calculator.ts         # §4 / §6
│   │   ├── bernoulli-heatmap.ts     # §6
│   │   └── fwd-vs-rev-fit.ts        # §7 — centerpiece
│   ├── ui/                          # shared chrome (see system spec)
│   │   ├── prereq-strip.ts
│   │   ├── crosslink-callout.ts
│   │   ├── nav-sidebar.ts
│   │   └── progress-bar.ts
│   └── styles/
│       ├── tokens.css               # imported from shared package
│       ├── prose.css
│       ├── components.css
│       └── viz.css
└── package.json
```

The `src/ui/` modules and `src/styles/tokens.css` are **shared with the EM
lesson and all future lessons** — they should live in a small shared
package or be symlinked. See `SYSTEM_AND_ROADMAP.md` for the contract.

---

## 2. Visual / Aesthetic Direction

Same paper-and-ink aesthetic as the EM lesson — same tokens, same
typography, same width rules, same drop caps. **Do not reinvent.** The
roadmap ties all lessons together visually; consistency is the entire
point.

Two **lesson-local accent colors** for this page (used for the two
distributions $p$ and $q$ throughout):

```css
--dist-p: #b8651a;   /* p — burnt sienna */
--dist-q: #1f6f8c;   /* q — deep teal     */
```

These replace the EM-lesson `--coin-a` / `--coin-b` for the duration of
this page. Other tokens (paper, ink, amber, sage) are unchanged.

---

## 3. Lesson Metadata (`src/meta.ts`)

Follows the schema in `SYSTEM_AND_ROADMAP.md` §2:

```ts
import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'kl-jensen',
  title: 'KL Divergence & Jensen\'s Inequality',
  subtitle: 'The two inequalities that hold up everything else.',
  tier: 1,                  // foundational
  difficulty: 2,            // 1–5
  estimatedHours: 3,
  status: 'planned',        // flip to 'built' on completion
  prerequisites: [],        // a strong undergrad math-stats background only
  recommendedNext: ['elbo-vi'],
  alsoUsedBy: ['em', 'elbo-vi', 'vae', 'ddpm'],
  description:
    'Jensen\'s inequality, KL divergence, the proof that KL ≥ 0, and the ' +
    'forward-vs-reverse-KL distinction that drives variational inference.',
  // sections that other lessons specifically link back to (anchors)
  exportedAnchors: {
    'jensen-statement': 'The statement of Jensen\'s inequality',
    'gibbs-inequality': 'Non-negativity of KL (Gibbs\' inequality)',
    'kl-gaussians':     'Closed form for KL between two Gaussians',
    'reverse-kl':       'Why VI uses reverse KL (mode-seeking)',
  },
};
```

The roadmap and other lessons consume this object.

---

## 4. Section-by-Section Plan

Eight sections. Aim for ~60 minutes of careful reading.

---

### Section 1 — Hook

**Length**: ~120 words. Full-width banner with a small SVG illustration of
two density curves overlapping (one in `--dist-p`, one in `--dist-q`).

**Prose** (verbatim):

> You've used Jensen's inequality once or twice in a homework problem.
> You've maybe seen KL divergence in passing — a "distance" between
> distributions, except it isn't a distance.
>
> What you might not yet appreciate is that *almost every modern result
> in probabilistic machine learning leans on these two ideas*. The
> convergence proof for EM uses them. The ELBO is built from them. The
> training objective for variational autoencoders is one of them. The
> loss function for diffusion models decomposes into a sum of them.
>
> This page earns the reader the right to use them. We'll prove
> Jensen's inequality, define KL divergence properly, prove that
> KL ≥ 0, and end with the picture that explains half of variational
> inference: **forward vs reverse KL**.

CTA button: "Let's start with convexity →"

---

### Section 2 — Convex and Concave Functions

**Length**: ~400 words. Foundational, mostly visual.

**Prose**:

> Before Jensen's inequality, we need to be precise about what *convex*
> means. A function $\varphi : \mathbb{R} \to \mathbb{R}$ is **convex**
> on an interval $I$ if for every $x_1, x_2 \in I$ and every
> $\lambda \in [0, 1]$,
>
> $$\varphi\big(\lambda x_1 + (1-\lambda) x_2\big) \;\leq\; \lambda \, \varphi(x_1) + (1-\lambda) \, \varphi(x_2)$$
>
> Geometrically: the **chord** between any two points on the graph lies
> on or above the curve. The function is **strictly convex** if the
> inequality is strict whenever $x_1 \neq x_2$ and $\lambda \in (0, 1)$
> — i.e., the chord lies *strictly* above the curve except at the
> endpoints.
>
> A function is **concave** if $-\varphi$ is convex; equivalently, the
> chord lies on or below the curve.
>
> Standard examples we'll use throughout the page:
>
> | Function | Domain | Convex / concave |
> |:---------|:------:|:----------------:|
> | $\varphi(x) = x^2$ | $\mathbb{R}$ | strictly convex |
> | $\varphi(x) = e^x$ | $\mathbb{R}$ | strictly convex |
> | $\varphi(x) = -\log(x)$ | $(0, \infty)$ | strictly convex |
> | $\varphi(x) = \log(x)$ | $(0, \infty)$ | strictly **concave** |
> | $\varphi(x) = |x|$ | $\mathbb{R}$ | convex (not strictly) |
>
> **Why we care about strict convexity**: it gives us *equality
> conditions* later. Jensen's inequality with a strictly convex
> $\varphi$ becomes an equality only when the random variable is
> degenerate (a constant). This will let us prove uniqueness results.
>
> If $\varphi$ is twice-differentiable, the test is simple: $\varphi$
> is convex iff $\varphi'' \geq 0$ everywhere; strictly convex iff
> $\varphi'' > 0$ everywhere.

**Callout — "log is the one to remember"** (`type=tip`):

> Throughout this page, the function we'll apply Jensen's inequality to
> is $\log$ (concave) — or equivalently $-\log$ (convex). Every KL
> property in this page reduces to: *carefully apply Jensen's
> inequality to a logarithm*. Internalize that and the rest is
> bookkeeping.

**Visualization 1 — `<ConvexExplorer>`** (medium width):

A 2D chart showing $y = \varphi(x)$ for a chosen $\varphi$. Two
draggable points $x_1, x_2$ on the x-axis with corresponding points on
the curve. A chord is drawn between $(x_1, \varphi(x_1))$ and
$(x_2, \varphi(x_2))$.

For each $\lambda$ (a third slider, default 0.5), display:
- The point $(x_\lambda, \varphi(x_\lambda))$ on the curve, where
  $x_\lambda = \lambda x_1 + (1-\lambda) x_2$
- The point $(x_\lambda, \lambda \varphi(x_1) + (1-\lambda) \varphi(x_2))$
  on the chord
- A vertical segment between them, labeled with its length

For convex $\varphi$, the chord point is above the curve point. For
concave $\varphi$, below. Toggle: a dropdown to pick $\varphi$ from
$\{x^2, e^x, -\log x, \log x, |x|, x^3\}$. Each pick relabels "convex"
or "concave" in a small badge in the corner.

---

### Section 3 — Jensen's Inequality

**Length**: ~600 words. The core theorem. Two proofs.

**Prose**:

> #### Statement
>
> Let $\varphi : \mathbb{R} \to \mathbb{R}$ be a convex function and let
> $X$ be a random variable with $\mathbb{E}|X| < \infty$ and
> $\mathbb{E}|\varphi(X)| < \infty$. Then
>
> $$\boxed{\;\; \varphi\big(\mathbb{E}[X]\big) \;\leq\; \mathbb{E}\big[\varphi(X)\big] \;\;}$$
>
> If $\varphi$ is concave, the inequality reverses. If $\varphi$ is
> *strictly* convex (resp. concave), equality holds if and only if $X$
> is a constant almost surely.
>
> Read it slowly. It says: **applying a convex function to an average
> is no bigger than averaging the function values**. Visually: replace
> "average" with "midpoint of a chord". The midpoint of a chord lies
> above the curve (convex). Done.

**Worked example (before the proofs)**:

> Take $\varphi(x) = x^2$ (convex) and any random variable $X$. Jensen
> gives:
>
> $$(\mathbb{E}[X])^2 \;\leq\; \mathbb{E}[X^2]$$
>
> Rearranging: $\mathbb{E}[X^2] - (\mathbb{E}[X])^2 \geq 0$, which is
> exactly $\mathrm{Var}(X) \geq 0$. **The non-negativity of variance is
> a one-line consequence of Jensen's inequality.**

> #### Proof for finitely many points
>
> Suppose $X$ takes values $x_1, \ldots, x_n$ with probabilities
> $p_1, \ldots, p_n$ summing to 1. We want:
>
> $$\varphi\Big(\sum_{i=1}^n p_i x_i\Big) \;\leq\; \sum_{i=1}^n p_i \, \varphi(x_i)$$
>
> Induction on $n$. The base case $n = 2$ is the definition of
> convexity with $\lambda = p_1$. For the inductive step, assume the
> claim for $n-1$ points. Let $\bar p = p_1 + \cdots + p_{n-1}$ (so
> $p_n = 1 - \bar p$). Define $\bar x = \tfrac{1}{\bar p} \sum_{i=1}^{n-1} p_i x_i$.
> Then by convexity ($n=2$ case applied to $\bar x$ and $x_n$ with
> weights $\bar p, 1 - \bar p$):
>
> $$\varphi\Big(\bar p \, \bar x + (1 - \bar p) x_n\Big) \;\leq\; \bar p \, \varphi(\bar x) + (1 - \bar p) \, \varphi(x_n)$$
>
> The left side is exactly $\varphi(\sum_i p_i x_i)$. By the inductive
> hypothesis applied to the renormalized $(p_i / \bar p)_{i=1}^{n-1}$:
>
> $$\varphi(\bar x) \;\leq\; \sum_{i=1}^{n-1} \frac{p_i}{\bar p} \varphi(x_i)$$
>
> Substituting and simplifying:
>
> $$\varphi\Big(\sum_i p_i x_i\Big) \;\leq\; \sum_{i=1}^{n-1} p_i \, \varphi(x_i) + (1 - \bar p) \varphi(x_n) \;=\; \sum_i p_i \, \varphi(x_i) \quad \blacksquare$$

> #### Proof for general random variables (supporting hyperplane)
>
> A cleaner proof that handles continuous distributions in one stroke.
>
> **Lemma** (supporting line). For any convex function $\varphi$ and
> any point $x_0$ in the interior of its domain, there exists a slope
> $a$ such that
>
> $$\varphi(x) \;\geq\; \varphi(x_0) + a \, (x - x_0) \quad \text{for all } x.$$
>
> (If $\varphi$ is differentiable at $x_0$, take $a = \varphi'(x_0)$.
> In general, $a$ is any element of the subdifferential — guaranteed
> non-empty for convex functions on the interior of their domain.)
>
> Now apply this with $x_0 = \mathbb{E}[X]$ and take expectation of
> both sides over $X$:
>
> $$\mathbb{E}[\varphi(X)] \;\geq\; \mathbb{E}\big[\varphi(\mathbb{E}[X]) + a(X - \mathbb{E}[X])\big] \;=\; \varphi(\mathbb{E}[X]) + a \cdot \mathbb{E}[X - \mathbb{E}[X]] \;=\; \varphi(\mathbb{E}[X])$$
>
> The cross-term vanishes because $\mathbb{E}[X - \mathbb{E}[X]] = 0$. $\blacksquare$
>
> This proof works for any random variable $X$ with $\mathbb{E}|X| < \infty$.

**Callout — "The proof you'll actually use"** (`type=tip`):

> The supporting-hyperplane proof is the one to remember. Every time
> you need Jensen, picture a tangent line to a convex curve at the
> mean: the curve is above the line everywhere; integrate; done.

**Visualization 2 — `<JensenGap>`** (full width):

A two-panel chart. Left panel: a chosen $\varphi$ (default $-\log$, so
the inequality reverses to $\log(\mathbb{E}[X]) \geq \mathbb{E}[\log X]$).
Right panel: a histogram of a chosen distribution of $X$ (Bernoulli /
Beta / Uniform / Gaussian / mixture, with sliders).

On the left panel, draw:
- The curve $\varphi$
- A vertical line at $x = \mathbb{E}[X]$
- A horizontal line at $y = \varphi(\mathbb{E}[X])$
- A horizontal line at $y = \mathbb{E}[\varphi(X)]$ (computed
  numerically from the histogram)
- The **gap** between them, labeled with its numeric value

As the user adjusts the distribution, the gap shrinks (when $X$ becomes
more peaked) or grows (when $X$ spreads). When $X$ is degenerate (a
spike), the gap goes to zero — *making the equality condition visceral*.

---

### Section 4 — KL Divergence: Definition

**Length**: ~500 words.

**Prose**:

> #### Definition
>
> Let $p$ and $q$ be probability distributions on the same space. The
> **Kullback–Leibler divergence** of $q$ from $p$ is
>
> $$\boxed{\;\; D_{\mathrm{KL}}(p \,\|\, q) \;:=\; \mathbb{E}_{X \sim p}\!\left[\log \frac{p(X)}{q(X)}\right] \;\;}$$
>
> In the discrete case:
>
> $$D_{\mathrm{KL}}(p \,\|\, q) \;=\; \sum_x p(x) \log \frac{p(x)}{q(x)}$$
>
> In the continuous case (with densities):
>
> $$D_{\mathrm{KL}}(p \,\|\, q) \;=\; \int p(x) \log \frac{p(x)}{q(x)} \, dx$$
>
> #### Conventions and edge cases
>
> Two annoying edge cases need conventions:
>
> 1. $0 \log 0 = 0$ (justified by $\lim_{t \to 0^+} t \log t = 0$). So
>    points where $p(x) = 0$ contribute nothing to the sum.
> 2. If there exists $x$ with $p(x) > 0$ and $q(x) = 0$, then
>    $D_{\mathrm{KL}}(p \,\|\, q) = +\infty$. This is more than just a
>    convention — it captures a *real fact*. If $q$ assigns zero
>    probability to something $p$ thinks is possible, no amount of
>    sampling from $q$ will ever produce that event, so the "code
>    based on $q$" cannot describe it. Infinity is correct.
>
> Formally: $D_{\mathrm{KL}}(p \,\|\, q) < \infty$ requires $p$ to be
> **absolutely continuous with respect to** $q$ — written $p \ll q$.
>
> #### Intuition: extra description length
>
> The cleanest interpretation comes from information theory. Suppose
> data is generated according to $p$, but we encode it using a code
> optimized for $q$ (assigning $-\log_2 q(x)$ bits to outcome $x$).
> The expected description length is $-\mathbb{E}_p[\log_2 q(X)]$. The
> *minimum possible* description length, achieved by the code optimized
> for $p$ itself, is $H(p) := -\mathbb{E}_p[\log_2 p(X)]$ (the entropy
> of $p$). The difference is:
>
> $$\underbrace{-\mathbb{E}_p[\log_2 q(X)]}_{\text{using $q$-code}} \;-\; \underbrace{-\mathbb{E}_p[\log_2 p(X)]}_{\text{using $p$-code (optimal)}} \;=\; \mathbb{E}_p\!\left[\log_2 \frac{p(X)}{q(X)}\right] \;=\; D_{\mathrm{KL}}(p \,\|\, q)$$
>
> So $D_{\mathrm{KL}}(p \,\|\, q)$ is **the expected number of extra
> bits required to describe $p$-distributed data using a $q$-optimized
> code**. (In machine learning we usually use natural log, so the unit
> is "nats" rather than "bits" — but the interpretation is identical.)

**Worked example: Bernoullis**:

> Let $p = \mathrm{Bern}(0.7)$ and $q = \mathrm{Bern}(0.5)$. Then
>
> $$D_{\mathrm{KL}}(p \,\|\, q) \;=\; 0.7 \log \frac{0.7}{0.5} + 0.3 \log \frac{0.3}{0.5} \;\approx\; 0.7 \cdot 0.336 + 0.3 \cdot (-0.511) \;\approx\; 0.0823 \text{ nats}$$
>
> Reverse direction:
>
> $$D_{\mathrm{KL}}(q \,\|\, p) \;=\; 0.5 \log \frac{0.5}{0.7} + 0.5 \log \frac{0.5}{0.3} \;\approx\; 0.0872 \text{ nats}$$
>
> Different! Already we see KL is **not symmetric**. We'll come back to
> this in §6.

**Worked example: Gaussians (full derivation)**:

> Let $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and
> $q = \mathcal{N}(\mu_2, \sigma_2^2)$. We derive the closed form
> because it's the building block of every Gaussian-latent variational
> objective (and hence of DDPM).
>
> Recall $\log \mathcal{N}(x; \mu, \sigma^2) = -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}$.
> So:
>
> $$\log \frac{p(x)}{q(x)} \;=\; \log \frac{\sigma_2}{\sigma_1} \;-\; \frac{(x - \mu_1)^2}{2\sigma_1^2} \;+\; \frac{(x - \mu_2)^2}{2\sigma_2^2}$$
>
> Take expectation under $X \sim p$. We need
> $\mathbb{E}_p[(X - \mu_1)^2] = \sigma_1^2$ (just the variance) and
> $\mathbb{E}_p[(X - \mu_2)^2] = \sigma_1^2 + (\mu_1 - \mu_2)^2$ (shift
> the mean and use the bias-variance decomposition). Substituting:
>
> $$D_{\mathrm{KL}}(p \,\|\, q) \;=\; \log\frac{\sigma_2}{\sigma_1} \;-\; \frac{\sigma_1^2}{2\sigma_1^2} \;+\; \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2}$$
>
> Simplifying:
>
> $$\boxed{\;\; D_{\mathrm{KL}}\!\big(\mathcal{N}(\mu_1, \sigma_1^2) \,\big\|\, \mathcal{N}(\mu_2, \sigma_2^2)\big) \;=\; \log \frac{\sigma_2}{\sigma_1} \;+\; \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} \;-\; \frac{1}{2} \;\;}$$
>
> **Sanity checks**: When $\mu_1 = \mu_2$ and $\sigma_1 = \sigma_2$,
> all three terms vanish — KL is zero, as it should be. When
> $\mu_1 = 0, \mu_2 = 1, \sigma_1 = \sigma_2 = 1$: $\log 1 + (1 + 1)/2 - 1/2 = \tfrac{1}{2}$.

This is anchor `kl-gaussians` — exported for future lessons.

**Visualization 3 — `<KLCalculator>`** (full width):

Two PMFs side-by-side, each over 6 outcomes labeled $\{1, 2, 3, 4, 5, 6\}$
(it's a "loaded die" framing). Each PMF has 6 sliders; the page enforces
sum-to-1 by renormalizing on each change.

Live readouts:
- $D_{\mathrm{KL}}(p \,\|\, q)$
- $D_{\mathrm{KL}}(q \,\|\, p)$
- Their ratio (to make asymmetry concrete)

A small bar chart below each readout decomposes the sum: shows
$p(x) \log(p(x)/q(x))$ for each outcome (positive bars when $p > q$ at
that outcome, negative when $p < q$). The bar values *sum to the KL*.
Hovering a bar highlights the corresponding outcome on both PMFs.

A "Try this" pre-set button: **set $q$ to zero on outcome 6** and watch
$D(p \| q) \to \infty$ (display "∞" with a small explanation).

---

### Section 5 — Non-Negativity (Gibbs' Inequality)

**Length**: ~400 words. **Anchor: `gibbs-inequality`** — exported for
EM and future lessons.

**Prose**:

> #### Theorem (Gibbs' inequality)
>
> For any two probability distributions $p, q$ on the same space,
>
> $$\boxed{\;\; D_{\mathrm{KL}}(p \,\|\, q) \;\geq\; 0 \;\;}$$
>
> with equality if and only if $p = q$ almost everywhere (with respect
> to $p$).
>
> This is the back-pocket inequality of probabilistic ML. Whenever a
> proof "needs an inequality," chances are this is the one.
>
> #### Proof
>
> By definition,
>
> $$D_{\mathrm{KL}}(p \,\|\, q) \;=\; \mathbb{E}_p\!\left[\log \frac{p(X)}{q(X)}\right] \;=\; -\mathbb{E}_p\!\left[\log \frac{q(X)}{p(X)}\right]$$
>
> Since $\log$ is **concave**, Jensen's inequality with $\varphi = \log$
> reverses, giving
>
> $$\mathbb{E}_p\!\left[\log \frac{q(X)}{p(X)}\right] \;\leq\; \log \mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right]$$
>
> Now compute the right-hand side directly:
>
> $$\mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] \;=\; \int p(x) \cdot \frac{q(x)}{p(x)} \, dx \;=\; \int q(x) \, dx \;=\; 1$$
>
> So $\mathbb{E}_p[\log(q/p)] \leq \log 1 = 0$, which gives
> $D_{\mathrm{KL}}(p \,\|\, q) = -\mathbb{E}_p[\log(q/p)] \geq 0$.
>
> For the **equality condition**: $\log$ is *strictly* concave, so
> Jensen is strict unless $q(X)/p(X)$ is constant almost surely under
> $p$. Combined with the constraint that both $p$ and $q$ integrate to
> 1, the constant must be 1, i.e. $p = q$ almost everywhere. $\blacksquare$

**Cross-link callout — back to EM** (`type=crosslink-back`):

> **Used by: [EM convergence theorem](../em-lesson/#monotonicity)**
>
> The EM convergence proof relied on the inequality
> $\mathrm{KL}(k(\cdot \mid x, \theta^{(t)}) \,\|\, k(\cdot \mid x, \theta^{(t+1)})) \geq 0$
> applied to the conditional distribution of the missing data. That
> inequality is exactly Gibbs' inequality, which we just proved. The
> EM lesson called it "Gibbs' inequality" and asked you to take it on
> trust. **It's no longer on trust.**

This is the first concrete cross-link in the system. It uses the
`<CrosslinkCallout>` component defined in `SYSTEM_AND_ROADMAP.md` §3.

**Worked example — sanity checks**:

> Verify on the Bernoulli example from §4: $D(\mathrm{Bern}(0.7) \| \mathrm{Bern}(0.5)) \approx 0.0823 > 0$
> and $D(\mathrm{Bern}(0.5) \| \mathrm{Bern}(0.7)) \approx 0.0872 > 0$.
> Both positive, both nonzero (since the distributions differ), as
> required.

**Visualization 4 — reuse `<KLCalculator>`** with a banner overlay
labeled "Verify: this number is always ≥ 0". Add a small running
minimum tracker that displays "lowest KL seen during this session: …"
to drive home that the user can never make it negative.

---

### Section 6 — Properties of KL

**Length**: ~500 words. Reference-card flavored.

**Prose**:

> KL has a handful of properties that get used constantly. We collect
> them here.
>
> #### 1. Asymmetry
>
> $D_{\mathrm{KL}}(p \,\|\, q) \neq D_{\mathrm{KL}}(q \,\|\, p)$ in
> general. Worked examples:
>
> | Comparison | $D(p \| q)$ | $D(q \| p)$ |
> |:-----------|:-----------:|:-----------:|
> | $\mathrm{Bern}(0.7) \;\|\; \mathrm{Bern}(0.5)$ | 0.0823 | 0.0872 |
> | $\mathrm{Bern}(0.9) \;\|\; \mathrm{Bern}(0.5)$ | 0.3681 | 0.5108 |
> | $\mathcal{N}(0, 1) \;\|\; \mathcal{N}(0, 4)$ | 0.318 | 0.807 |
>
> Notice the third row: when $q$ is much wider than $p$,
> $D(p \| q) < D(q \| p)$. Why? Because $D(q \| p)$ has $q$'s broad
> mass landing in $p$'s tails, where $\log(q/p)$ is large; whereas
> $D(p \| q)$ has $p$'s narrow mass on a region where $q$ is
> reasonably accurate.
>
> #### 2. Not a metric
>
> Beyond asymmetry, KL also fails the triangle inequality. Counterexamples
> are easy to construct on three Bernoullis. So calling it a "distance"
> is a useful intuition pump but a formal lie.
>
> #### 3. Convex in the pair
>
> $(p, q) \mapsto D_{\mathrm{KL}}(p \,\|\, q)$ is jointly convex.
> Specifically, for any $\lambda \in [0, 1]$ and pairs $(p_1, q_1),
> (p_2, q_2)$,
>
> $$D_{\mathrm{KL}}\!\big(\lambda p_1 + (1-\lambda) p_2 \,\big\|\, \lambda q_1 + (1-\lambda) q_2\big) \;\leq\; \lambda \, D_{\mathrm{KL}}(p_1 \,\|\, q_1) + (1-\lambda) \, D_{\mathrm{KL}}(p_2 \,\|\, q_2)$$
>
> (Proof: log-sum inequality. We won't reproduce it here — it's
> tedious — but it's worth knowing the name.)
>
> #### 4. Pinsker's inequality
>
> KL upper-bounds total variation:
>
> $$\| p - q \|_{\mathrm{TV}} \;\leq\; \sqrt{\tfrac{1}{2} D_{\mathrm{KL}}(p \,\|\, q)}$$
>
> So when KL is small, $p$ and $q$ are *close* in total variation
> distance — they assign similar probabilities to every event. Pinsker
> is what makes KL useful as a *practical* distance even though it isn't
> formally one.
>
> #### 5. Chain rule
>
> For joint distributions $p(x, y) = p(x) p(y \mid x)$ and analogously
> for $q$,
>
> $$D_{\mathrm{KL}}(p(x, y) \,\|\, q(x, y)) \;=\; D_{\mathrm{KL}}(p(x) \,\|\, q(x)) \;+\; \mathbb{E}_{X \sim p}\!\left[D_{\mathrm{KL}}\!\big(p(y \mid X) \,\|\, q(y \mid X)\big)\right]$$
>
> This is the analogue of the chain rule for entropy. We'll use it in
> the DDPM lesson, where the KL across a Markov chain decomposes step
> by step.

**Visualization 5 — `<BernoulliHeatmap>`** (medium width):

A 2D heatmap of $D_{\mathrm{KL}}(\mathrm{Bern}(p) \,\|\, \mathrm{Bern}(q))$
over $(p, q) \in (0.01, 0.99)^2$. Color via a sequential colormap (e.g.
`viridis`). The diagonal $p = q$ is dark blue (KL = 0). Far corners
$(0.01, 0.99)$ and $(0.99, 0.01)$ are the brightest (KL is large).

A toggle button: **"Show $D(q \| p)$ instead"** flips the heatmap.
Side-by-side, the two heatmaps look *visibly different* — same diagonal,
but the off-diagonal asymmetry is the punchline.

Hover any pixel to see numerical $(p, q, D)$ in a tooltip. Click to
freeze a marker; click again to clear. Useful for verifying the table
above.

---

### Section 7 — Forward vs Reverse KL

**Length**: ~600 words. **Anchor: `reverse-kl`** — exported for the
ELBO and VI lesson, which is essentially built on this section's
visualization.

**Prose**:

> Here's the picture that explains half of variational inference.
>
> Suppose we have a complicated **target** distribution $p$ and we want
> to **approximate** it with a simpler distribution $q$ chosen from
> some tractable family $\mathcal{Q}$ (e.g., $q = \mathcal{N}(\mu, \sigma^2)$
> with $\mu, \sigma$ to be chosen). Two natural objectives:
>
> $$q^{\text{fwd}} \;:=\; \arg\min_{q \in \mathcal{Q}} \, D_{\mathrm{KL}}(p \,\|\, q) \qquad \text{("forward KL")}$$
>
> $$q^{\text{rev}} \;:=\; \arg\min_{q \in \mathcal{Q}} \, D_{\mathrm{KL}}(q \,\|\, p) \qquad \text{("reverse KL")}$$
>
> These give *very different* answers. Intuitively:
>
> - **Forward KL is mass-covering.** It penalizes $q(x)$ being **small**
>   wherever $p(x)$ is **large** (because $\log(p/q)$ blows up there).
>   So $q$ stretches to cover the full support of $p$ — it would
>   rather over-extend than miss anything.
>
> - **Reverse KL is mode-seeking.** It penalizes $q(x)$ being **large**
>   wherever $p(x)$ is **small** (because $\log(q/p)$ blows up there).
>   So $q$ contracts onto a region where $p$ has substantial mass — it
>   would rather miss some of $p$'s modes than place mass where $p$ has
>   none.
>
> The cleanest demonstration: **target $p$ is a bimodal mixture, family
> $\mathcal{Q}$ is unimodal Gaussians**.

**Worked example — bimodal target**:

> Let $p(x) = \tfrac{1}{2} \mathcal{N}(x; -3, 1) + \tfrac{1}{2} \mathcal{N}(x; 3, 1)$
> and let $\mathcal{Q} = \{ \mathcal{N}(\mu, \sigma^2) \}$.
>
> **Forward KL minimum.** Working through the algebra (or noting that
> minimizing $D(p \| q)$ over a Gaussian family is moment-matching to
> $p$): the optimum is $q^{\text{fwd}} = \mathcal{N}(\mathbb{E}_p[X], \mathrm{Var}_p[X])$.
> For the symmetric mixture, $\mathbb{E}_p[X] = 0$ and
> $\mathrm{Var}_p[X] = \tfrac{1}{2}(1 + 9) + \tfrac{1}{2}(1 + 9) - 0 = 10$.
> So $q^{\text{fwd}} = \mathcal{N}(0, 10)$ — wide, centered between the
> two modes, mass everywhere $p$ has any.
>
> **Reverse KL minimum.** Numerical optimization (no closed form)
> finds $q^{\text{rev}} \approx \mathcal{N}(\pm 3, 1.05)$ — by symmetry
> there are two optima, each sitting tightly on one mode and ignoring
> the other.

**Picture this** (referenced from the visualization below): forward KL
gives a single fat Gaussian straddling both modes; reverse KL gives a
narrow Gaussian sitting on one mode. **Same target, different
objectives, completely different fits.**

**Cross-link callout — forward to ELBO** (`type=crosslink-forward`):

> **Used by: [ELBO & Variational Inference](../elbo-lesson/) (planned)**
>
> Variational inference uses **reverse KL**: it picks $q$ to minimize
> $D_{\mathrm{KL}}(q \,\|\, p_{\text{posterior}})$. The mode-seeking
> behavior is exactly why VI sometimes underestimates posterior
> uncertainty — it commits to one mode and ignores others. The next
> lesson will show that minimizing reverse KL is equivalent to
> maximizing the **evidence lower bound (ELBO)**, the workhorse
> objective of modern probabilistic ML.

**Visualization 6 — `<ForwardVsReverseFit>`** (full width, ≥600px tall):

The centerpiece of the page. Three coordinated panels:

```
┌─────────────────────────────────────────────────────────┐
│  TARGET  p(x)                                            │
│  Sliders: mode separation Δ, mode weight π, mode width σ │
│  Default: 0.5·N(-3,1) + 0.5·N(3,1)                       │
└─────────────────────────────────────────────────────────┘
┌────────────────────────────┬────────────────────────────┐
│  FORWARD KL FIT             │  REVERSE KL FIT             │
│  q minimizing D(p‖q)        │  q minimizing D(q‖p)        │
│                             │                             │
│  [chart: p in burnt sienna, │  [chart: p in burnt sienna, │
│   q^fwd in deep teal,       │   q^rev in deep teal,       │
│   filled q overlay]         │   filled q overlay]         │
│                             │                             │
│  μ = 0.00,  σ² = 10.00      │  μ = 3.00,  σ² = 1.05       │
│  D(p‖q) = 1.16              │  D(q‖p) = 0.69              │
│                             │                             │
│  "Mass-covering: q stretches│  "Mode-seeking: q sits on    │
│   to cover both modes"      │   one mode, ignores other"   │
└────────────────────────────┴────────────────────────────┘
```

Behavior:

- When the user adjusts the target sliders, **both fits recompute live**
  via gradient descent on the two KLs (50 steps from a sensible
  initialization). Forward KL has a closed-form moment-match; reverse KL
  needs numerical optimization.
- Buttons under each panel: **"Watch the fit converge"** — animates the
  optimization trajectory frame-by-frame so the user can see the
  optimizer settle.
- A **"Reverse KL: try other init"** button starts reverse KL from
  $\mu = -3$ instead of $\mu = +3$ — converges to the *other* mode.
  Demonstrates the multi-modal optimum landscape of reverse KL.
- A small text annotation pops up when the modes get close together
  (separation < 1.5σ): "When the modes are close, forward and reverse
  KL agree." Important pedagogical point.

This visualization is worth substantial implementation effort. It is
the single most-cited image when explaining VI.

---

### Section 8 — Where You'll See This

**Length**: ~250 words. The cross-page navigation hub.

**Prose**:

> Two inequalities and one divergence. You'll see them everywhere in
> probabilistic ML; here's a curated list of where, with links into the
> lessons that build on them.
>
> #### Already used (look back)
>
> - **EM convergence theorem.** The proof needed
>   $\mathrm{KL}(k(\cdot \mid x, \theta^{(t)}) \,\|\, k(\cdot \mid x, \theta^{(t+1)})) \geq 0$.
>   That's Gibbs' inequality (§5).
>   → [Revisit the EM convergence proof](../em-lesson/#monotonicity)
>
> #### Coming next
>
> - **ELBO & Variational Inference.** Built directly from a
>   reverse-KL objective (§7). The "evidence lower bound" is what you
>   get when you rearrange $D_{\mathrm{KL}}(q \,\|\, p_{\text{posterior}})$.
>   → Next lesson on the roadmap.
>
> - **Variational Autoencoders.** Loss = reconstruction term + KL
>   regularizer. The KL is between the encoder $q(z \mid x)$ and a
>   prior $p(z) = \mathcal{N}(0, I)$, evaluated using the
>   Gaussian–Gaussian closed form derived in §4.
>
> - **DDPM (the destination).** The training objective is a sum of KL
>   terms, each between two Gaussians along a Markov chain. The chain
>   rule of KL (§6) breaks it into per-step pieces; the Gaussian KL
>   formula (§4) makes each piece tractable.
>
> #### Adjacent ideas (sidebar lessons)
>
> - **Cross-entropy in classification.** Cross-entropy is just
>   $H(p) + D_{\mathrm{KL}}(p \,\|\, q)$; minimizing cross-entropy =
>   minimizing forward KL.
>
> - **Maximum likelihood estimation.** MLE minimizes
>   $D_{\mathrm{KL}}(\widehat p_{\text{data}} \,\|\, p_\theta)$ — the
>   forward KL between the empirical and model distributions. Every
>   ML textbook says this in passing; we now know exactly what it
>   means.

**Visualization 7 — `<RoadmapMini>`** (medium width):

A small embedded view of the roadmap (see `SYSTEM_AND_ROADMAP.md` §4),
with this lesson highlighted as "you are here" and arrows showing
incoming dependencies (none) and outgoing dependencies (EM, ELBO, VAE,
DDPM, all dimmed-or-bright depending on built status). Click any node
to navigate.

---

## 5. Algorithm / Math Implementation

### `src/math/kl.ts`

```ts
/** KL between two discrete distributions (arrays of equal length, each summing to 1). */
export function klDiscrete(p: number[], q: number[]): number {
  if (p.length !== q.length) throw new Error('p and q must have same length');
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] === 0) continue;             // 0 log 0 = 0
    if (q[i] === 0) return Infinity;      // p > 0, q = 0
    s += p[i] * Math.log(p[i] / q[i]);
  }
  return s;
}

/** KL between two Bernoullis. */
export function klBernoulli(p: number, q: number): number {
  return klDiscrete([p, 1 - p], [q, 1 - q]);
}

/** Closed-form KL between two univariate Gaussians. */
export function klGaussian(mu1: number, sigma1: number,
                           mu2: number, sigma2: number): number {
  return Math.log(sigma2 / sigma1)
       + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
       - 0.5;
}

/** Numeric KL between two continuous densities sampled on a uniform grid. */
export function klContinuous(p: number[], q: number[], dx: number): number {
  if (p.length !== q.length) throw new Error('p and q must have same length');
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] < 1e-12) continue;
    if (q[i] < 1e-12) return Infinity;
    s += p[i] * Math.log(p[i] / q[i]) * dx;
  }
  return s;
}
```

### `src/math/jensen.ts`

```ts
/** Numerically compute the Jensen gap φ(E[X]) vs E[φ(X)] for a discrete distribution. */
export function jensenGap(values: number[], probs: number[],
                          phi: (x: number) => number): {
  expectationOfPhi: number;
  phiOfExpectation: number;
  gap: number;
} {
  let ex = 0, ephi = 0;
  for (let i = 0; i < values.length; i++) {
    ex   += values[i] * probs[i];
    ephi += phi(values[i]) * probs[i];
  }
  const phiOfEx = phi(ex);
  return {
    expectationOfPhi: ephi,
    phiOfExpectation: phiOfEx,
    gap: ephi - phiOfEx,    // ≥ 0 for convex φ; ≤ 0 for concave
  };
}
```

### Test cases (`kl.test.ts`)

- `klBernoulli(0.7, 0.5)` ≈ 0.0823
- `klBernoulli(0.5, 0.7)` ≈ 0.0872
- `klBernoulli(0.5, 0.5)` === 0
- `klGaussian(0, 1, 1, 1)` === 0.5
- `klGaussian(0, 1, 0, 2)` ≈ 0.3181
- `klGaussian(0, 2, 0, 1)` ≈ 0.8069
- `klDiscrete([0.5, 0.5], [0.5, 0.5])` === 0
- `klDiscrete([0.5, 0.5], [1, 0])` === Infinity
- For any `p, q`, `klDiscrete(p, q) >= 0` (random-input fuzz test, 100 cases)

---

## 6. Component Catalog

### Shared (live in the cross-lesson UI package — see system spec)
- `<NavigationSidebar>`, `<ProgressBar>` — same as EM lesson.
- `<PrereqStrip>` — strip at the top of the page listing prerequisite
  lessons with status badges (built / planned).
- `<CrosslinkCallout type="back|forward|sidebar">` — colored callout
  with arrow icon indicating direction of cross-page reference.

### Lesson-local
- `<MathInline>`, `<MathBlock>`, `<Callout>`, `<DataTable>`,
  `<ProofToggle>` — same primitives as EM lesson.

### Visualizations
- `<ConvexExplorer>` (§2)
- `<JensenGap>` (§3)
- `<KLCalculator>` (§4, reused in §5)
- `<BernoulliHeatmap>` (§6)
- `<ForwardVsReverseFit>` (§7) — centerpiece
- `<RoadmapMini>` (§8)

The `<RoadmapMini>` is a *shared* component pulled from the roadmap page
(see `SYSTEM_AND_ROADMAP.md` §4) — it should be a thin adapter that
takes the catalog and a `currentLessonId`.

---

## 7. Page-Level UX

Same as EM:

- Sticky left sidebar with §1–§8.
- Top progress bar.
- Keyboard: ←/→ for sections, Space to step the active visualization
  optimizer (in §7).
- Reduced-motion media queries honored on the §3 and §7 animations.
- Dark mode supported via the same token system.

**New** for this lesson (added by the cross-page system):

- A `<PrereqStrip>` at the very top, just below the title. Empty
  prereqs (this is foundational) → strip says "Foundational lesson —
  no prerequisites. ✨ Recommended starting point."
- A `<CrosslinkCallout>` in §5 pointing back to the EM convergence
  proof.
- A `<CrosslinkCallout>` in §7 pointing forward to the planned
  ELBO/VI lesson.
- A footer `<RoadmapMini>` showing this lesson's position in the
  graph.

---

## 8. Acceptance Criteria

A learner who has worked through this page should be able to, on a blank
sheet:

1. State Jensen's inequality for both convex and concave $\varphi$,
   including the equality conditions.
2. Prove Jensen's inequality using the supporting-hyperplane argument
   (one paragraph).
3. Write the definition of KL divergence (discrete and continuous
   forms) and explain the convention for $p > 0, q = 0$.
4. Prove $D_{\mathrm{KL}}(p \,\|\, q) \geq 0$ in three lines, citing
   Jensen.
5. Derive $D_{\mathrm{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \,\|\, \mathcal{N}(\mu_2, \sigma_2^2))$
   from the definition.
6. Explain — without invoking formulas — why forward KL is
   mass-covering and reverse KL is mode-seeking, in terms of which
   ratio blows up where.
7. Identify which result from this page is invoked in the EM
   convergence proof.

If a friendly TA quizzed the learner on the above and they failed two
or more, the page didn't do its job.

---

## 9. Stretch Goals (post-MVP)

- **Coding interpretation widget**: a small toy interactive that
  simulates encoding 100 samples from $p$ using a code optimized for
  $q$, and compares the average bits/symbol to entropy + KL. Concrete
  payoff for the "extra description length" intuition in §4.
- **Pinsker visualization**: plot
  $\| p - q \|_{\mathrm{TV}}$ vs $\sqrt{D_{\mathrm{KL}}(p \,\|\, q) / 2}$
  for a sweep of distribution pairs; show Pinsker's bound holds.
- **f-divergence sidebar**: a brief mention that KL is a special case
  of an $f$-divergence, with a chart showing other common ones (TV,
  $\chi^2$, Hellinger). Out of scope for the main flow.
- **Animated derivation of the Gaussian KL**: each algebraic step
  appears one at a time on a "Next" button click.

---

## 10. Out of Scope (intentionally)

- Multivariate Gaussian KL (in full generality, with covariance
  matrices). Defer to the ELBO/VI or VAE lesson, where it'll be
  needed concretely.
- $f$-divergences as a unified framework (only mentioned).
- Information geometry beyond a name-drop.
- Pinsker's *proof* (statement only).
- Renyi divergences, Wasserstein distances, MMD — all great topics,
  none of them on the path to DDPM.