# Gaussian Cookbook — Interactive Reference
## Build Specification & Content Plan

> A reference page collecting the four Gaussian identities that the
> VAE, DDPM, and Score Matching lessons depend on:
>
> 1. KL divergence between two multivariate Gaussians.
> 2. The reparameterization trick in matrix form.
> 3. Conditioning and marginalization of jointly Gaussian variables.
> 4. The linear-Gaussian Bayesian update.
>
> Unlike the other lessons, this one is a **cookbook**, not a narrative.
> A reader doesn't read it cover-to-cover; they land here from a
> deep-link in another lesson, look up the identity they need, and
> leave. Each section is self-contained, each identity has a worked
> numerical example, and the navigation supports look-up rather than
> sequential reading.
>
> **Position in the roadmap**: foundational (tier 1). Required for
> VAE and DDPM. Builds on KL & Jensen (which covered the univariate
> Gaussian KL).

---

## 0. Pedagogical Philosophy

This page deviates slightly from the other lessons:

1. **Look-up first, read-through second.** The page is optimized for
   landing from an external deep-link to a specific identity. The
   sidebar always shows section anchors prominently; each section
   begins with the **formula in a boxed equation** before any prose.
2. **Derivations are still complete, but collapsible.** Each identity
   has its derivation behind a `<ProofToggle>` (collapsed by default —
   the formula is what most readers want; the proof is for when they
   want to verify). The toggle is more aggressive here than in
   narrative lessons.
3. **Every identity has a numerical example.** The reader can plug in
   numbers, watch the formula produce the right answer, and trust it.
4. **Cross-links forward dominate.** This page exists to be linked to.
   Each section ends with a "used by" callout listing where the
   identity matters downstream.

By the end, the reader can: (a) compute KL between two multivariate
Gaussians using the closed form, (b) apply the reparameterization
trick in matrix form to sample from a parametric Gaussian
differentiably, (c) compute the conditional distribution of one
component of a jointly Gaussian vector given the other, (d) compute
the posterior in a linear-Gaussian model.

---

## 1. Tech Stack

Same as other lessons. Lives in `src/lessons/gaussian-cookbook/` under
the StatViz monorepo. Imports `klGaussian` and shared chrome from the
established packages.

File layout:

```
src/lessons/gaussian-cookbook/
├── main.ts
├── meta.ts
├── math/
│   ├── mvn.ts                   # multivariate Gaussian density, sampling
│   ├── kl-mvn.ts                # KL between multivariate Gaussians
│   ├── reparam.ts               # the reparameterization trick
│   ├── conditioning.ts          # conditional & marginal
│   ├── linear-gaussian.ts       # the Bayesian update
│   └── *.test.ts
├── sections/
│   ├── 01-hook.ts
│   ├── 02-mvn-foundations.ts
│   ├── 03-kl-multivariate.ts
│   ├── 04-reparameterization.ts
│   ├── 05-conditioning.ts
│   ├── 06-linear-gaussian-bayes.ts
│   └── 07-where-youll-see-this.ts
├── viz/
│   ├── mvn-explorer.ts          # §2 — 2D Gaussian with adjustable Sigma
│   ├── kl-mvn-explorer.ts       # §3 — two 2D Gaussians, live KL
│   ├── reparam-flow.ts          # §4 — epsilon → z visualization
│   ├── conditioning-slice.ts    # §5 — joint to conditional, sliced
│   └── linear-gauss-bayes.ts    # §6 — prior, likelihood, posterior
└── styles/
    └── overrides.css
```

For matrix operations: use a lightweight linear-algebra utility. The
repo already depends on D3 (which doesn't ship matrix ops). Add a
small dependency: `ml-matrix` (≈30 kB, supports inv, det, Cholesky,
eigendecomp). This is the **only** new dependency this lesson
introduces; flag it for review during the agent's first plan
report-back.

---

## 2. Visual / Aesthetic Direction

Same paper-and-ink aesthetic. Lesson-local accents:

```css
--gauss-p:    #b8651a;   /* "p" distribution (matches dist-p in KL) */
--gauss-q:    #1f6f8c;   /* "q" distribution (matches dist-q in KL) */
--gauss-grid: #d4ccbf;   /* contour grid lines */
--accent-ev:  #2c5f8d;   /* eigenvector arrows in MVN explorer */
```

The `--gauss-p` and `--gauss-q` colors match the KL lesson's
distribution colors intentionally — the visual continuity reinforces
that this is "the multivariate sequel."

---

## 3. Lesson Metadata (`src/lessons/gaussian-cookbook/meta.ts`)

```ts
import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'gaussian-cookbook',
  title: 'Gaussian Cookbook',
  subtitle: 'The four Gaussian identities you\'ll keep using.',
  tier: 1,
  difficulty: 2,
  estimatedHours: 2,
  status: 'planned',
  prerequisites: [
    {
      id: 'kl-jensen',
      strength: 'required',
      anchor: 'kl-gaussians',
    },
  ],
  recommendedNext: ['vae', 'ddpm'],
  alsoUsedBy: ['vae', 'ddpm', 'score-matching'],
  description:
    'Four Gaussian identities collected as a reference: multivariate ' +
    'KL, the reparameterization trick in matrix form, conditioning ' +
    'and marginalization of jointly Gaussian variables, and the ' +
    'linear-Gaussian Bayesian update.',
  exportedAnchors: {
    'mvn-density':       'The multivariate Gaussian density',
    'kl-mvn':            'KL divergence between multivariate Gaussians',
    'kl-mvn-diag':       'KL when both covariances are diagonal (VAE regularizer)',
    'reparam-matrix':    'The reparameterization trick (matrix form)',
    'conditioning':      'Conditional & marginal of jointly Gaussian variables',
    'linear-gauss-bayes':'Linear-Gaussian Bayesian update (closed-form posterior)',
  },
  path: '/lessons/gaussian-cookbook',
};
```

---

## 4. Section-by-Section Plan

Seven sections. Total reading time: ~60 minutes if read cover-to-cover,
but designed for ~5 minutes per identity look-up.

---

### Section 1 — Hook

**Length**: ~120 words. Brief because this is a reference.

**Prose** (verbatim):

> Most of the math in VAE, DDPM, and score-based models reduces to
> manipulating Gaussians. Specifically, four identities show up over
> and over:
>
> - The **KL between two Gaussians** — closed form, used everywhere.
> - The **reparameterization trick** — how to backpropagate through
>   a Gaussian sample.
> - **Conditioning** — given a joint Gaussian over $(X, Y)$, what's
>   the distribution of $Y$ given $X = x$?
> - **Linear-Gaussian Bayes** — given a Gaussian prior and a Gaussian
>   likelihood with a linear mean, what's the posterior?
>
> This page collects them. Each is stated, derived, and demonstrated
> with numbers you can verify. The page is designed to be **looked up**
> — a reference, not a story. Jump to whichever identity you need.

A `<JumpToTable>` component below the prose: a grid of four chips,
each linking to the relevant section. Tappable shortcut to bypass the
foundations if the reader already knows the basics.

---

### Section 2 — The Multivariate Gaussian: foundations

**Length**: ~350 words. **Anchor: `mvn-density`**.

**Prose**:

> A $d$-dimensional random vector $X \in \mathbb{R}^d$ is **multivariate
> Gaussian** with mean $\mu \in \mathbb{R}^d$ and covariance
> $\Sigma \in \mathbb{R}^{d \times d}$ (symmetric positive definite)
> if its density is
>
> $$\boxed{\;\; \mathcal{N}(x; \mu, \Sigma) \;=\; \frac{1}{(2\pi)^{d/2} \, |\Sigma|^{1/2}} \, \exp\!\left(-\tfrac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)\right) \;\;}$$
>
> where $|\Sigma|$ is the determinant. The **log density** drops the
> exponential and the constants:
>
> $$\log \mathcal{N}(x; \mu, \Sigma) \;=\; -\tfrac{d}{2} \log(2\pi) \;-\; \tfrac{1}{2} \log |\Sigma| \;-\; \tfrac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)$$
>
> The quadratic form $(x - \mu)^\top \Sigma^{-1} (x - \mu)$ is the
> **squared Mahalanobis distance** between $x$ and $\mu$ under the
> metric $\Sigma^{-1}$.
>
> #### Three special cases worth memorizing
>
> 1. **Isotropic** ($\Sigma = \sigma^2 I$): contours are spheres,
>    density depends only on $\|x - \mu\|^2$:
>
>    $$\mathcal{N}(x; \mu, \sigma^2 I) \;\propto\; \exp\!\left(-\frac{\|x - \mu\|^2}{2\sigma^2}\right)$$
>
> 2. **Diagonal** ($\Sigma = \mathrm{diag}(\sigma_1^2, \ldots, \sigma_d^2)$):
>    coordinates are independent, density factorizes:
>
>    $$\mathcal{N}(x; \mu, \Sigma) \;=\; \prod_{i=1}^{d} \mathcal{N}(x_i; \mu_i, \sigma_i^2)$$
>
> 3. **Standard** ($\mu = 0, \Sigma = I$): the reference distribution,
>    used as the prior in nearly every VAE and as the noise distribution
>    in DDPM.

> #### Affine transformations preserve Gaussianity
>
> A useful fact: if $X \sim \mathcal{N}(\mu, \Sigma)$ and we form
> $Y = AX + b$ for $A \in \mathbb{R}^{m \times d}$ and $b \in \mathbb{R}^m$,
> then
>
> $$Y \;\sim\; \mathcal{N}(A\mu + b, \; A \Sigma A^\top)$$
>
> The means transform linearly; the covariance transforms as a
> bilinear form. This is the structural fact that makes the
> reparameterization trick (§4) work.

**Visualization 1 — `<MVNExplorer>`** (full width, medium height):

A 2D Gaussian density visualization. The user can:
- Drag the mean $\mu$ around the canvas.
- Adjust three sliders: $\Sigma_{11}, \Sigma_{22}, \Sigma_{12}$ (with
  $\Sigma_{21} = \Sigma_{12}$, and `\Sigma` validated to be positive
  definite — show a red warning otherwise).
- Toggle "show contours" / "show samples" / "show eigenvectors".

Display shows:
- Density via a contour plot (3 ellipses at 1, 2, 3 standard deviations).
- 200 random samples (using the reparameterization trick under the
  hood — same code that powers §4).
- Two arrows showing the eigenvectors of $\Sigma$ scaled by $\sqrt{\lambda_i}$
  (the principal axes of the covariance ellipse).

Side panel readouts:
- $\mu = (\mu_1, \mu_2)$
- $\Sigma = \begin{pmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{12} & \sigma_{22}\end{pmatrix}$
- $\det \Sigma$
- Eigenvalues $\lambda_1, \lambda_2$ (and correlation $\rho$ for intuition)

This is the foundational visualization — readers internalize what
$\Sigma$ does before any algebra.

---

### Section 3 — KL Divergence Between Multivariate Gaussians

**Length**: ~600 words. **Anchors: `kl-mvn`, `kl-mvn-diag`**.

**The formula box, before any prose**:

> $$\boxed{\;\; D_{\mathrm{KL}}\!\big(\mathcal{N}(\mu_1, \Sigma_1) \,\big\|\, \mathcal{N}(\mu_2, \Sigma_2)\big) \;=\; \tfrac{1}{2}\!\left[\log \frac{|\Sigma_2|}{|\Sigma_1|} \;-\; d \;+\; \mathrm{tr}(\Sigma_2^{-1} \Sigma_1) \;+\; (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1)\right] \;\;}$$

**Prose**:

> The univariate version (from [KL & Jensen §4](../kl-jensen/#kl-gaussians))
> had three terms: a log-ratio of standard deviations, a sum involving
> variance and mean-squared-distance, and a $-\tfrac{1}{2}$. The
> multivariate version has the same four terms generalized:
>
> - $\log |\Sigma_2| / |\Sigma_1|$ — log determinant ratio (the
>   multivariate analogue of $\log \sigma_2/\sigma_1$).
> - $-d$ — the dimension (analogue of the $-\tfrac{1}{2}$ × 2).
> - $\mathrm{tr}(\Sigma_2^{-1} \Sigma_1)$ — a "size ratio" term.
> - $(\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1)$ — squared
>   Mahalanobis distance between means under $\Sigma_2^{-1}$.
>
> All terms are scalars. The overall expression is non-negative (by
> Gibbs' inequality from the KL lesson), and zero iff
> $(\mu_1, \Sigma_1) = (\mu_2, \Sigma_2)$.

#### Derivation (wrapped in `<ProofToggle>`, collapsed by default)

> Start from the definition:
>
> $$D_{\mathrm{KL}} \;=\; \mathbb{E}_{X \sim \mathcal{N}(\mu_1, \Sigma_1)}\!\left[\log \frac{\mathcal{N}(X; \mu_1, \Sigma_1)}{\mathcal{N}(X; \mu_2, \Sigma_2)}\right]$$
>
> Use the log density formula from §2:
>
> $$\log \frac{\mathcal{N}(x; \mu_1, \Sigma_1)}{\mathcal{N}(x; \mu_2, \Sigma_2)} \;=\; \tfrac{1}{2}\log \frac{|\Sigma_2|}{|\Sigma_1|} \;-\; \tfrac{1}{2}(x - \mu_1)^\top \Sigma_1^{-1}(x - \mu_1) \;+\; \tfrac{1}{2}(x - \mu_2)^\top \Sigma_2^{-1}(x - \mu_2)$$
>
> Take expectation under $X \sim \mathcal{N}(\mu_1, \Sigma_1)$. The
> log-determinant term is constant. For the two quadratic forms, use:
>
> $$\mathbb{E}[(X - a)^\top M (X - a)] \;=\; \mathrm{tr}(M \Sigma_1) \;+\; (\mu_1 - a)^\top M (\mu_1 - a)$$
>
> (Easy to verify by expanding $X - a = (X - \mu_1) + (\mu_1 - a)$ and
> using linearity.)
>
> Applying to the first quadratic with $a = \mu_1, M = \Sigma_1^{-1}$:
>
> $$\mathbb{E}[(X - \mu_1)^\top \Sigma_1^{-1}(X - \mu_1)] \;=\; \mathrm{tr}(\Sigma_1^{-1} \Sigma_1) + 0 \;=\; d$$
>
> Applying to the second with $a = \mu_2, M = \Sigma_2^{-1}$:
>
> $$\mathbb{E}[(X - \mu_2)^\top \Sigma_2^{-1}(X - \mu_2)] \;=\; \mathrm{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_1 - \mu_2)^\top \Sigma_2^{-1}(\mu_1 - \mu_2)$$
>
> Combining:
>
> $$D_{\mathrm{KL}} \;=\; \tfrac{1}{2}\log \frac{|\Sigma_2|}{|\Sigma_1|} - \tfrac{1}{2} d + \tfrac{1}{2}\mathrm{tr}(\Sigma_2^{-1}\Sigma_1) + \tfrac{1}{2}(\mu_2 - \mu_1)^\top \Sigma_2^{-1}(\mu_2 - \mu_1)$$
>
> Pulling out the $\tfrac{1}{2}$ gives the boxed formula. $\blacksquare$

#### Two important special cases

**Diagonal covariances** (**anchor: `kl-mvn-diag`**) — this is the VAE
regularizer:

> If $\Sigma_1 = \mathrm{diag}(\sigma_{1,1}^2, \ldots, \sigma_{1,d}^2)$
> and $\Sigma_2 = \mathrm{diag}(\sigma_{2,1}^2, \ldots, \sigma_{2,d}^2)$,
> the formula simplifies to a sum of $d$ univariate KLs:
>
> $$D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1, \Sigma_1) \,\big\|\, \mathcal{N}(\mu_2, \Sigma_2)\big) \;=\; \tfrac{1}{2}\sum_{i=1}^{d}\!\left[\log \frac{\sigma_{2,i}^2}{\sigma_{1,i}^2} \;-\; 1 \;+\; \frac{\sigma_{1,i}^2 + (\mu_{1,i} - \mu_{2,i})^2}{\sigma_{2,i}^2}\right]$$
>
> **VAE special case**: $q = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$,
> $p = \mathcal{N}(0, I)$:
>
> $$\boxed{\;\; D_{\mathrm{KL}}\big(q \,\|\, \mathcal{N}(0, I)\big) \;=\; \tfrac{1}{2}\sum_{i=1}^{d}\left[\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2\right] \;\;}$$
>
> Memorize this. Every VAE implementation has it as a one-line
> contribution to the loss.

**Shared covariance**:

> If $\Sigma_1 = \Sigma_2 = \Sigma$, the log-det, trace-trace, and $-d$
> terms collapse:
>
> $$D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1, \Sigma) \,\big\|\, \mathcal{N}(\mu_2, \Sigma)\big) \;=\; \tfrac{1}{2}(\mu_1 - \mu_2)^\top \Sigma^{-1} (\mu_1 - \mu_2)$$
>
> Just half the squared Mahalanobis distance between means. **DDPM
> uses this:** the per-timestep KL between two Gaussians sharing the
> same forward-process variance reduces to this clean form.

#### Worked numerical examples

| $p$ | $q$ | $D_{\mathrm{KL}}(p \| q)$ |
|:----|:----|:-----------------------:|
| $\mathcal{N}([1, 0, -1], \mathrm{diag}(1, 2, 1))$ | $\mathcal{N}(0, I)$ | $1.1534$ |
| $\mathcal{N}([0.5, -0.2], \mathrm{diag}(e^{0.2}, e^{-0.6}))$ | $\mathcal{N}(0, I)$ | $0.2301$ |
| $\mathcal{N}([1, 1], \begin{pmatrix}1 & 0.5\\0.5 & 1\end{pmatrix})$ | $\mathcal{N}([0, 0], 2I)$ | $0.8370$ |
| (same swap) | $D_{\mathrm{KL}}(q\|p)$ | $1.4963$ |

(All values pre-verified; verify in `kl-mvn.test.ts`.)

**Visualization 2 — `<KLMVNExplorer>`** (full width):

Two 2D Gaussians overlaid on a single canvas (`--gauss-p` and
`--gauss-q`). Each has draggable mean and four sliders for the
covariance matrix entries. Live readouts:
- $D_{\mathrm{KL}}(p \| q)$
- $D_{\mathrm{KL}}(q \| p)$
- The four sub-terms of $D_{\mathrm{KL}}(p \| q)$ broken out (log-det,
  $-d$, trace, Mahalanobis) so the reader can see which contributes
  most.

Buttons:
- "Set $q$ to $\mathcal{N}(0, I)$" — the VAE special case.
- "Match means" — set $\mu_q = \mu_p$. Watch only the Mahalanobis term
  go to zero; the others remain.
- "Match covariances" — set $\Sigma_q = \Sigma_p$. Watch the log-det,
  $-d$, and trace cancel; only Mahalanobis survives.

**Cross-link callout — forward** (`type=forward`):

> **Comes back in:** [VAE](../vae/) — the regularizer term in the VAE
> loss is this KL with $q = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x)))$
> and $p = \mathcal{N}(0, I)$. [DDPM](../ddpm/) — every per-timestep
> loss is this KL with shared covariance.

---

### Section 4 — The Reparameterization Trick (Matrix Form)

**Length**: ~400 words. **Anchor: `reparam-matrix`**.

**The formula box first**:

> $$\boxed{\;\; Z \sim \mathcal{N}(\mu, \Sigma) \;\iff\; Z = \mu + L \, \varepsilon, \;\; \varepsilon \sim \mathcal{N}(0, I), \;\; L L^\top = \Sigma \;\;}$$

**Prose**:

> To sample $Z \sim \mathcal{N}(\mu, \Sigma)$ in a way that's
> **differentiable with respect to $\mu$ and $\Sigma$**, factor $\Sigma$
> as $L L^\top$ (where $L$ is, e.g., the Cholesky factor), draw a
> standard normal $\varepsilon \sim \mathcal{N}(0, I)$, and form
>
> $$Z \;=\; \mu + L \varepsilon$$
>
> Why this works: $\varepsilon$ has zero mean and identity covariance,
> so by the affine-transformation rule from §2,
>
> $$\mathbb{E}[Z] = \mu, \quad \mathrm{Cov}(Z) = L \cdot I \cdot L^\top = \Sigma$$
>
> $Z$ is Gaussian (affine of a Gaussian), with the right mean and
> covariance, so $Z \sim \mathcal{N}(\mu, \Sigma)$.
>
> #### Why the trick matters
>
> The point isn't to sample (which we could do directly). The point
> is that the **randomness has been separated from the parameters**.
> $\varepsilon$ has no parameters; $\mu$ and $L$ are deterministic
> functions of whatever upstream variables we care about. So for any
> smooth $f$:
>
> $$\nabla_{\mu, L} \, \mathbb{E}_{Z \sim \mathcal{N}(\mu, \Sigma)}[f(Z)] \;=\; \nabla_{\mu, L} \, \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I)}[f(\mu + L\varepsilon)] \;=\; \mathbb{E}_{\varepsilon}\!\left[\nabla_{\mu, L} f(\mu + L \varepsilon)\right]$$
>
> Gradients pass through the sampling step. **This is what makes
> end-to-end gradient training of VAEs possible.**

#### Diagonal special case (the one VAEs use)

> If $\Sigma = \mathrm{diag}(\sigma_1^2, \ldots, \sigma_d^2)$, then
> $L = \mathrm{diag}(\sigma_1, \ldots, \sigma_d)$ and the rule
> collapses to element-wise:
>
> $$Z_i \;=\; \mu_i + \sigma_i \varepsilon_i, \;\; \varepsilon_i \sim \mathcal{N}(0, 1)$$
>
> In vector notation: $Z = \mu + \sigma \odot \varepsilon$ where
> $\odot$ is element-wise multiplication. **One line of code in any
> VAE implementation.**

#### Worked numerical example

> Take $\mu = (1, -1)$ and $\Sigma = \begin{pmatrix} 4 & 1 \\ 1 & 1 \end{pmatrix}$.
> The Cholesky factor is
>
> $$L \;=\; \begin{pmatrix} 2 & 0 \\ 0.5 & \sqrt{0.75} \end{pmatrix} \;\approx\; \begin{pmatrix} 2 & 0 \\ 0.5 & 0.866 \end{pmatrix}$$
>
> Drawing $\varepsilon = (0.3, -1.2)$ gives
>
> $$Z = (1, -1) + (2 \cdot 0.3, \; 0.5 \cdot 0.3 + 0.866 \cdot (-1.2)) = (1.6, -1.890)$$
>
> The transformation is deterministic given $\varepsilon$; gradients
> with respect to $\mu$ and $L$ pass straight through.

**Visualization 3 — `<ReparamFlow>`** (full width):

Two side-by-side 2D canvases:
- Left: 200 samples of $\varepsilon \sim \mathcal{N}(0, I)$ as a cloud
  of dots.
- Right: the same 200 samples after transformation $Z = \mu + L\varepsilon$,
  with the contour ellipse of $\mathcal{N}(\mu, \Sigma)$ overlaid.

Each sample on the left is **paired** to its transformed counterpart
on the right via a faint line. Hover a sample to highlight the pair.
The user adjusts:
- $\mu$ via dragging (on the right canvas).
- Three sliders for $\Sigma_{11}, \Sigma_{22}, \Sigma_{12}$.

As the sliders move, watch the right cloud **morph deterministically**
from the left cloud — the same $\varepsilon$ samples, just under a
different affine map. **The pedagogical point**: the noise is fixed;
the parameters do all the work. That's why gradients can flow.

A button "Re-roll $\varepsilon$" generates new noise. Same right-side
distribution, different sample positions.

**Cross-link callout — forward** (`type=forward`):

> **Comes back in:** [VAE](../vae/) — the encoder outputs
> $(\mu_\phi(x), \sigma_\phi^2(x))$; the sample $z = \mu + \sigma \odot \varepsilon$
> is differentiable end-to-end through the encoder. [DDPM](../ddpm/) —
> noise is added via $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \varepsilon$,
> which is the reparameterization trick.

---

### Section 5 — Conditioning and Marginalization

**Length**: ~600 words. **Anchor: `conditioning`**.

**The formula box first**:

> Let $(X, Y) \sim \mathcal{N}\!\left(\begin{pmatrix}\mu_X \\ \mu_Y\end{pmatrix}, \begin{pmatrix}\Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_{YY}\end{pmatrix}\right)$
> be jointly Gaussian. Then:
>
> $$\boxed{\;\; X \;\sim\; \mathcal{N}(\mu_X, \Sigma_{XX}) \;\;}$$
>
> $$\boxed{\;\; Y \mid X = x \;\sim\; \mathcal{N}\!\big(\mu_Y + \Sigma_{YX} \Sigma_{XX}^{-1} (x - \mu_X), \;\; \Sigma_{YY} - \Sigma_{YX} \Sigma_{XX}^{-1} \Sigma_{XY}\big) \;\;}$$

**Prose**:

> A jointly Gaussian distribution stays Gaussian under both
> **marginalization** (drop $Y$) and **conditioning** (fix $X$). The
> marginal is easy — just read off the relevant block of the mean and
> covariance. The conditional is the powerful identity; let's
> understand it.
>
> #### Reading the conditional mean
>
> $$\mu_{Y \mid X = x} \;=\; \mu_Y + \Sigma_{YX} \Sigma_{XX}^{-1}(x - \mu_X)$$
>
> Start at $\mu_Y$ and shift by an amount proportional to how much $x$
> deviates from $\mu_X$. The "amount" is governed by $\Sigma_{YX}
> \Sigma_{XX}^{-1}$ — the **regression coefficient** of $Y$ on $X$.
>
> When $X$ and $Y$ are uncorrelated ($\Sigma_{YX} = 0$), the shift is
> zero: knowing $X$ tells you nothing about $Y$, so $\mu_{Y|X} = \mu_Y$.
> When they're highly correlated, the shift is large.
>
> #### Reading the conditional covariance
>
> $$\Sigma_{Y \mid X} \;=\; \Sigma_{YY} - \Sigma_{YX} \Sigma_{XX}^{-1} \Sigma_{XY}$$
>
> Start at the marginal covariance of $Y$ and **subtract** the
> variance "explained" by $X$ (the Schur complement of $\Sigma_{XX}$).
> Conditioning on $X$ never increases uncertainty about $Y$: the
> subtracted term is always positive semidefinite.

#### Derivation (wrapped in `<ProofToggle>`)

> Strategy: complete the square in $y$ inside the joint density.
> Easier with block matrix manipulation. We'll use the **block matrix
> inversion** formula. Write
>
> $$\Sigma^{-1} \;=\; \begin{pmatrix} \Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_{YY} \end{pmatrix}^{\!-1}$$
>
> Standard result (Schur complement):
>
> $$\Sigma^{-1} \;=\; \begin{pmatrix} \Sigma_{XX}^{-1} + \Sigma_{XX}^{-1}\Sigma_{XY} S^{-1} \Sigma_{YX}\Sigma_{XX}^{-1} & -\Sigma_{XX}^{-1}\Sigma_{XY} S^{-1} \\ -S^{-1} \Sigma_{YX}\Sigma_{XX}^{-1} & S^{-1} \end{pmatrix}$$
>
> where $S = \Sigma_{YY} - \Sigma_{YX} \Sigma_{XX}^{-1}\Sigma_{XY}$ is
> the **Schur complement of $\Sigma_{XX}$** — the thing inside the
> boxed conditional covariance.
>
> The joint log-density has $\propto -\tfrac{1}{2}\begin{pmatrix}x - \mu_X \\ y - \mu_Y\end{pmatrix}^\top \Sigma^{-1}\begin{pmatrix}x - \mu_X \\ y - \mu_Y\end{pmatrix}$
> as the only term involving $y$. Expanding the bottom-right block:
>
> $$\propto -\tfrac{1}{2}(y - \mu_Y)^\top S^{-1}(y - \mu_Y) \;+\; (y - \mu_Y)^\top S^{-1} \Sigma_{YX}\Sigma_{XX}^{-1}(x - \mu_X) \;+\; (\text{constant in } y)$$
>
> Completing the square in $y$ around the point
> $\mu_Y + \Sigma_{YX}\Sigma_{XX}^{-1}(x - \mu_X)$ gives a Gaussian in
> $y$ with covariance $S$ and that mean — matching the boxed
> identity. $\blacksquare$

#### Worked numerical example

> Take $(X, Y) \sim \mathcal{N}\!\left(\begin{pmatrix}0 \\ 0\end{pmatrix}, \begin{pmatrix}1 & 0.7 \\ 0.7 & 1\end{pmatrix}\right)$
> and observe $X = 1$. Then:
>
> $$\mu_{Y \mid X = 1} \;=\; 0 + 0.7 \cdot 1 \cdot (1 - 0) \;=\; 0.7$$
>
> $$\sigma_{Y \mid X}^{\,2} \;=\; 1 - 0.7 \cdot 1 \cdot 0.7 \;=\; 0.51$$
>
> So $Y \mid X = 1 \sim \mathcal{N}(0.7, 0.51)$. The mean is **pulled
> toward** $X$ (correlation 0.7 → 70% of the deviation passes through),
> and the conditional variance (0.51) is **smaller** than the marginal
> (1.0) because knowing $X$ has resolved some uncertainty about $Y$.

**Visualization 4 — `<ConditioningSlice>`** (full width):

A 2D scatter plot of $(X, Y) \sim \mathcal{N}(\mu, \Sigma)$ with the
joint density as a heatmap behind 500 sampled points. A vertical
dashed line at the current $x$-value (user-draggable slider). Below
the scatter:
- A 1D Gaussian curve showing $Y \mid X = x$ — the conditional density.
- Its mean and variance updating live.

Sliders for $\Sigma$ entries on the side. As correlation $\rho$
increases (slide $\Sigma_{12}$ up), the conditional curve **tracks**
the slider position more aggressively and **narrows** (variance
shrinks). At $\rho = 0$, the conditional is just the marginal.

**Cross-link callout — forward** (`type=forward`):

> **Comes back in:** [DDPM](../ddpm/) — the analytical posterior
> $q(z_{t-1} \mid z_t, x_0)$ that the score network's target. The
> forward process $(z_0, z_1, \ldots, z_T)$ is jointly Gaussian; this
> identity gives the conditional in closed form.

---

### Section 6 — The Linear-Gaussian Bayesian Update

**Length**: ~500 words. **Anchor: `linear-gauss-bayes`**.

**The formula box first**:

> Given:
> - Prior: $z \sim \mathcal{N}(\mu_0, \Sigma_0)$
> - Likelihood: $x \mid z \sim \mathcal{N}(A z + b, \Sigma_n)$
>
> The posterior is Gaussian:
>
> $$\boxed{\;\; z \mid x \;\sim\; \mathcal{N}(\mu_{\text{post}}, \Sigma_{\text{post}}) \;\;}$$
>
> with
>
> $$\Sigma_{\text{post}} \;=\; \big(\Sigma_0^{-1} + A^\top \Sigma_n^{-1} A\big)^{-1}$$
>
> $$\mu_{\text{post}} \;=\; \Sigma_{\text{post}}\!\left[\Sigma_0^{-1} \mu_0 + A^\top \Sigma_n^{-1} (x - b)\right]$$

**Prose**:

> This is the workhorse identity for closed-form Bayesian inference
> under Gaussian assumptions. The prior and the likelihood are both
> Gaussian, the likelihood has a linear mean in $z$, and the posterior
> falls out Gaussian.
>
> #### Reading the formulas
>
> **Posterior precision** = prior precision + likelihood precision:
>
> $$\Sigma_{\text{post}}^{-1} \;=\; \Sigma_0^{-1} \;+\; A^\top \Sigma_n^{-1} A$$
>
> Precisions add. The observation contributes an "evidence
> precision" $A^\top \Sigma_n^{-1} A$ proportional to how informative
> the linear map $A$ is.
>
> **Posterior mean** = precision-weighted combination of prior mean
> and observation:
>
> $$\Sigma_{\text{post}}^{-1} \mu_{\text{post}} \;=\; \Sigma_0^{-1} \mu_0 \;+\; A^\top \Sigma_n^{-1} (x - b)$$
>
> Stronger likelihood (smaller $\Sigma_n$) pulls $\mu_{\text{post}}$
> toward $A^{-1}(x - b)$. Stronger prior (smaller $\Sigma_0$) pulls
> $\mu_{\text{post}}$ toward $\mu_0$.

#### Derivation (wrapped in `<ProofToggle>`)

> Two routes — pick whichever is more transparent.
>
> **Route 1 (algebraic):** Multiply prior and likelihood, complete the
> square in $z$:
>
> $$\log p(z \mid x) \;=\; \log p(x \mid z) + \log p(z) + \text{const}$$
>
> $$\propto -\tfrac{1}{2}(x - Az - b)^\top \Sigma_n^{-1}(x - Az - b) \;-\; \tfrac{1}{2}(z - \mu_0)^\top \Sigma_0^{-1}(z - \mu_0)$$
>
> Expanding and collecting terms in $z$, the quadratic coefficient is
> $\tfrac{1}{2}(A^\top \Sigma_n^{-1} A + \Sigma_0^{-1})$, so
> $\Sigma_{\text{post}}^{-1} = A^\top \Sigma_n^{-1} A + \Sigma_0^{-1}$.
> The linear coefficient gives $\Sigma_{\text{post}}^{-1} \mu_{\text{post}}$
> as in the box.
>
> **Route 2 (conditioning):** $(z, x)$ is jointly Gaussian (because
> $z$ is Gaussian and $x = Az + b + \mathrm{noise}$, an affine
> transformation plus Gaussian noise). Apply §5's conditioning
> formula. After some algebra, the same answer falls out — but it
> takes the right block-matrix manipulation to make it clean. $\blacksquare$

#### Worked numerical example

> Take a 2D prior and a noisy linear observation:
>
> - Prior: $z \sim \mathcal{N}(0, I_2)$
> - Likelihood: $x \mid z \sim \mathcal{N}\!\left(\begin{pmatrix}1 & 0.5 \\ 0.3 & 1\end{pmatrix} z, \;\; 0.1 \, I_2\right)$
> - Observe $x = (1.5, 0.8)$
>
> Computing:
>
> $$A^\top \Sigma_n^{-1} A \;=\; 10 \begin{pmatrix}1 & 0.3 \\ 0.5 & 1\end{pmatrix}\begin{pmatrix}1 & 0.5 \\ 0.3 & 1\end{pmatrix} \;=\; 10 \begin{pmatrix}1.09 & 0.8 \\ 0.8 & 1.25\end{pmatrix}$$
>
> $$\Sigma_{\text{post}}^{-1} \;=\; I + 10\begin{pmatrix}1.09 & 0.8 \\ 0.8 & 1.25\end{pmatrix} \;=\; \begin{pmatrix}11.9 & 8 \\ 8 & 13.5\end{pmatrix}$$
>
> $$\Sigma_{\text{post}} \;\approx\; \begin{pmatrix}0.1397 & -0.0828 \\ -0.0828 & 0.1231\end{pmatrix}, \quad \mu_{\text{post}} \;\approx\; (1.147, \; 0.468)$$
>
> (Verify in `linear-gaussian.test.ts`. Numbers pre-computed.)

**Visualization 5 — `<LinearGaussianBayes>`** (full width):

Three 2D Gaussians overlaid on a single canvas:
- Prior $\mathcal{N}(\mu_0, \Sigma_0)$ in `--gauss-q`.
- Posterior $\mathcal{N}(\mu_{\text{post}}, \Sigma_{\text{post}})$ in
  `--gauss-p`.
- Likelihood "contour" — a line/curve showing where in $z$-space the
  observation $x$ is most consistent (the locus $Az + b = x$, or its
  Gaussian thickening by $\Sigma_n$).

User controls: $A$ entries, $\Sigma_n$ entries, prior covariance, the
observed $x$. As parameters change, the posterior morphs live. Watch:
- Lowering $\Sigma_n$ (sharper likelihood) shrinks the posterior toward
  the likelihood curve.
- Tightening the prior (smaller $\Sigma_0$) pulls posterior toward the
  prior mean.
- Singular $A$ (rank-deficient) — only some directions get constrained,
  the rest match the prior.

**Cross-link callout — forward** (`type=forward`):

> **Comes back in:** Kalman filtering (sidebar topic; not built),
> Gaussian process regression (sidebar topic; not built). Both are
> repeated applications of this formula.

---

### Section 7 — Where You'll See This

**Length**: ~200 words.

**Prose**:

> Four identities. Here's where each shows up in lessons you may
> already have read, or will read next.
>
> #### Outgoing references (these lessons depend on this one)
>
> - **VAE** — uses [§3 diagonal KL](#kl-mvn-diag) for the regularizer
>   term in the loss; uses [§4 reparameterization](#reparam-matrix)
>   to backprop through the encoder's sample.
> - **DDPM** — uses [§3 shared-covariance KL](#kl-mvn) for the
>   per-timestep loss; uses [§5 conditioning](#conditioning) to
>   derive the analytical posterior $q(z_{t-1} \mid z_t, x_0)$;
>   uses [§4 reparameterization](#reparam-matrix) to sample noise
>   along the diffusion chain.
> - **Score Matching** — uses [§4 reparameterization](#reparam-matrix)
>   to construct noisy data; uses the score of a Gaussian (which
>   falls out of §2).
>
> #### Incoming references (this lesson depends on these)
>
> - **KL & Jensen** — established the univariate Gaussian KL in §4;
>   §3 here is its multivariate generalization. The non-negativity
>   property used implicitly here is also from there.
>
> #### Sidebar topics that lean on the same identities
>
> - **Kalman filtering** — repeated linear-Gaussian Bayes (§6).
> - **Gaussian processes** — conditioning on observations (§5).
> - **Linear-Gaussian state-space models** — combine §5 and §6.

**Visualization 7 — `<RoadmapMini>`** showing this lesson connecting
to VAE, DDPM, Score Matching (all planned at this point), with KL &
Jensen as the only incoming dependency.

---

## 5. Algorithm / Math Implementation

### `src/lessons/gaussian-cookbook/math/mvn.ts`

```ts
import { Matrix, determinant, inverse, CholeskyDecomposition } from 'ml-matrix';

/** Log density of x under N(mu, Sigma). */
export function logMVNDensity(x: number[], mu: number[], Sigma: number[][]): number {
  const d = mu.length;
  const SigmaM = new Matrix(Sigma);
  const det = determinant(SigmaM);
  const inv = inverse(SigmaM);
  const diff = x.map((xi, i) => xi - mu[i]);
  const diffM = Matrix.columnVector(diff);
  const quad = diffM.transpose().mmul(inv).mmul(diffM).get(0, 0);
  return -0.5 * d * Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * quad;
}

/** Sample one realization from N(mu, Sigma) via reparameterization. */
export function sampleMVN(mu: number[], Sigma: number[][], rng: () => number = Math.random): number[] {
  const d = mu.length;
  const SigmaM = new Matrix(Sigma);
  const chol = new CholeskyDecomposition(SigmaM);
  const L = chol.lowerTriangularMatrix;
  const eps: number[] = [];
  for (let i = 0; i < d; i++) eps.push(boxMuller(rng));
  const epsM = Matrix.columnVector(eps);
  const Leps = L.mmul(epsM).to1DArray();
  return mu.map((m, i) => m + Leps[i]);
}

function boxMuller(rng: () => number): number {
  let u = rng(); while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
```

### `src/lessons/gaussian-cookbook/math/kl-mvn.ts`

```ts
import { Matrix, determinant, inverse, trace } from 'ml-matrix';

/** KL(N(mu1, Sigma1) || N(mu2, Sigma2)). */
export function klMVN(mu1: number[], Sigma1: number[][], mu2: number[], Sigma2: number[][]): number {
  const d = mu1.length;
  const S1 = new Matrix(Sigma1);
  const S2 = new Matrix(Sigma2);
  const S2inv = inverse(S2);
  const det1 = determinant(S1);
  const det2 = determinant(S2);
  const diff = Matrix.columnVector(mu1.map((m, i) => mu2[i] - m));
  const maha = diff.transpose().mmul(S2inv).mmul(diff).get(0, 0);
  const tr = trace(S2inv.mmul(S1));
  return 0.5 * (Math.log(det2 / det1) - d + tr + maha);
}

/** Diagonal-Gaussian-against-standard-normal: VAE regularizer. */
export function klDiagFromStandard(mu: number[], log_sigma: number[]): number {
  let s = 0;
  for (let i = 0; i < mu.length; i++) {
    const sigma2 = Math.exp(2 * log_sigma[i]);
    s += sigma2 + mu[i]*mu[i] - 1 - 2*log_sigma[i];
  }
  return 0.5 * s;
}
```

### `src/lessons/gaussian-cookbook/math/conditioning.ts`

```ts
import { Matrix, inverse } from 'ml-matrix';

/** Given jointly Gaussian (X, Y) with mu = [mu_X; mu_Y] and the block covariance,
 *  compute Y | X = x as (mu_cond, Sigma_cond). */
export function conditionalGaussian(
  mu_X: number[], mu_Y: number[],
  Sigma_XX: number[][], Sigma_XY: number[][],
  Sigma_YX: number[][], Sigma_YY: number[][],
  x: number[],
): { mu: number[]; Sigma: number[][] } {
  const SXX_inv = inverse(new Matrix(Sigma_XX));
  const SYX = new Matrix(Sigma_YX);
  const SXY = new Matrix(Sigma_XY);
  const SYY = new Matrix(Sigma_YY);
  const xDev = Matrix.columnVector(x.map((xi, i) => xi - mu_X[i]));

  // mu_cond = mu_Y + Sigma_YX Sigma_XX^-1 (x - mu_X)
  const shift = SYX.mmul(SXX_inv).mmul(xDev).to1DArray();
  const mu_cond = mu_Y.map((m, i) => m + shift[i]);

  // Sigma_cond = Sigma_YY - Sigma_YX Sigma_XX^-1 Sigma_XY
  const Sigma_cond = SYY.sub(SYX.mmul(SXX_inv).mmul(SXY)).to2DArray();
  return { mu: mu_cond, Sigma: Sigma_cond };
}
```

### `src/lessons/gaussian-cookbook/math/linear-gaussian.ts`

```ts
import { Matrix, inverse } from 'ml-matrix';

/** Given prior z ~ N(mu_0, Sigma_0) and likelihood x | z ~ N(A z + b, Sigma_n),
 *  compute posterior z | x = N(mu_post, Sigma_post). */
export function linearGaussianPosterior(
  mu_0: number[], Sigma_0: number[][],
  A: number[][], b: number[], Sigma_n: number[][],
  x: number[],
): { mu: number[]; Sigma: number[][] } {
  const S0_inv = inverse(new Matrix(Sigma_0));
  const Sn_inv = inverse(new Matrix(Sigma_n));
  const AM = new Matrix(A);
  const xb = Matrix.columnVector(x.map((xi, i) => xi - b[i]));

  const precision = S0_inv.add(AM.transpose().mmul(Sn_inv).mmul(AM));
  const Sigma_post = inverse(precision);
  const mu_post_v = Sigma_post
    .mmul(S0_inv.mmul(Matrix.columnVector(mu_0)).add(AM.transpose().mmul(Sn_inv).mmul(xb)))
    .to1DArray();
  return { mu: mu_post_v, Sigma: Sigma_post.to2DArray() };
}
```

### Test cases

- `klMVN([1, 0, -1], diag([1,2,1]), [0,0,0], eye(3))` ≈ 1.1534
- `klDiagFromStandard([0.5, -0.2], [0.1, -0.3])` ≈ 0.2301
- `klMVN([1, 1], [[1,0.5],[0.5,1]], [0,0], [[2,0],[0,2]])` ≈ 0.8370
- `klMVN([0, 0], [[2,0],[0,2]], [1, 1], [[1,0.5],[0.5,1]])` ≈ 1.4963
- Conditioning example: `(0.7, 0.51)` for $Y | X=1$.
- Linear-Gaussian example: $\mu_{\text{post}} \approx (1.1474, 0.4682)$,
  $\Sigma_{\text{post}}[0,0] \approx 0.1397$, $\Sigma_{\text{post}}[1,1] \approx 0.1231$.

---

## 6. Component Catalog

### Shared (already exist)
Standard chrome from `@shared/ui/`.

### Lesson-local visualizations
- `<MVNExplorer>` (§2)
- `<KLMVNExplorer>` (§3)
- `<ReparamFlow>` (§4)
- `<ConditioningSlice>` (§5)
- `<LinearGaussianBayes>` (§6)
- `<JumpToTable>` (§1) — small navigation aid

---

## 7. Page-Level UX

Same as the other lessons. Three notes specific to this lesson:

1. **`<JumpToTable>` at the top.** Below the §1 prose, render a grid
   of four chips (one per identity), each linking to the relevant
   anchor. Lets readers bypass §2 if they don't need the foundations
   refresher.

2. **Section anchors prominent in URL bar.** When a reader scrolls
   into a section, update the URL hash via History API. This
   supports the use case: "I read DDPM, it linked me to
   `#conditioning`, I read that, I copy the URL to share with a
   colleague."

3. **`<ProofToggle>` collapsed by default everywhere.** Unlike the
   narrative lessons (EM, KL, ELBO) where proofs are inline by
   default, here they hide. The reader has a formula to use; the
   derivation is a click away when they want it.

---

## 8. Acceptance Criteria

A reader who has worked through this page should be able to, on a
blank sheet:

1. Write the multivariate Gaussian density formula.
2. State and apply the diagonal-Gaussian KL formula (the VAE
   regularizer special case) from memory.
3. Explain why the reparameterization trick makes gradients flow,
   in two sentences.
4. Given a $2 \times 2$ joint covariance and an observed $x$,
   compute $Y \mid X = x$ by hand (mean and variance).
5. Set up the linear-Gaussian posterior for a 1D toy problem and
   solve for $\mu_{\text{post}}, \sigma_{\text{post}}^2$.

A reader who has just landed on a specific section via deep-link
should be able to: read the boxed formula, verify it against the
worked example, and leave with the formula in 5 minutes or less.

---

## 9. Stretch Goals (post-MVP)

- **Block diagonal special case** of conditioning when $\Sigma_{XY} = 0$
  (independence). Cute. Skip for now.
- **Generalized eigenvalue decomposition** for visualizing covariance
  ellipses with confidence radii. The §2 viz uses this implicitly;
  could expose it.
- **Whitening transform**: $W = \Sigma^{-1/2}$ such that $WX$ has
  identity covariance. Mention in §4; full treatment as stretch.
- **The Woodbury matrix identity** as a sidebar in §6 — used to
  compute posterior covariance more efficiently when one of the
  matrices is high-rank.

---

## 10. Out of Scope (intentionally)

- **Matrix calculus identities** beyond what's needed. Petersen &
  Pedersen's matrix cookbook is the right reference; don't reproduce.
- **Numerical issues** beyond "always work in log-space for densities"
  and "use Cholesky, not direct inverse, for sampling." A deeper
  treatment belongs in a numerics-focused page.
- **Infinite-dimensional Gaussians (Gaussian processes)** — separate
  sidebar lesson, not here.
- **Matrix-variate Gaussians** — adjacent and not on the DDPM path.