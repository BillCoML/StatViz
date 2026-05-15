# Score Matching — Interactive Lesson
## Build Specification & Content Plan

> A deep, interactive lesson on score-based generative modeling: an
> alternative approach to fitting $p(x)$ that sidesteps the
> normalization constant entirely by modeling $\nabla_x \log p(x)$
> directly. By the end, the reader knows what a score function is,
> three ways to fit one (Fisher divergence directly is intractable;
> implicit and denoising score matching are the tractable surrogates),
> how to sample via Langevin dynamics, and why annealing noise levels
> is the bridge that leads directly into DDPM.
>
> **Position in the roadmap**: applications tier (tier 3). Parallel
> branch to VAE — both feed into DDPM. Builds on KL & Jensen (for
> the Fisher-divergence framing) and Gaussian Cookbook (the score of
> a Gaussian + noise corruption + reparameterization all live there).
> Does **not** depend on ELBO/VI or VAE — score-based models are
> genuinely a different paradigm.

---

## 0. Pedagogical Philosophy

Same commitments as the rest of StatViz:

1. **Concrete before abstract.** The lesson opens with "what would
   it mean to model the gradient of the log-density instead of the
   density itself?" before any formula appears.
2. **The integration-by-parts trick is shown in full.** §4's
   Hyvärinen derivation is the conceptually-hardest step of the
   lesson. We don't skip it — we walk through it slowly, with the
   boundary assumption stated explicitly.
3. **The Vincent identity is the centerpiece equation.** §5 culminates
   in the denoising score matching loss, which is literally the DDPM
   training objective. Everything in §5 builds toward making that
   identity feel inevitable.
4. **Visualizations show vector fields, not just curves.** Score
   functions are vector fields on $\mathbb{R}^d$. The 2D
   visualizations all show flow / quiver plots. Particles moving
   under Langevin dynamics make the "score field as wind" intuition
   physical.

By the end, the reader can: (a) define the score function and explain
why it's invariant to the normalization constant, (b) state and
derive both implicit and denoising score matching losses, (c) explain
why denoising avoids the trace-of-Jacobian cost, (d) state the
Langevin sampling algorithm and explain why noise is necessary,
(e) describe how annealing connects single-level score matching to
DDPM's multi-step training.

---

## 1. Tech Stack

Same Vite multi-page setup. One new consideration:

- **Tiny score model.** §7's annealed-Langevin visualization runs a
  pre-trained score network $s_\theta(x, \sigma)$ in the browser.
  Architecture is small enough that vanilla TS matrix multiplication
  (via `ml-matrix`, already installed) handles inference. Pre-train
  in Python, ship weights as JSON.

File layout:

```
src/lessons/score-matching/
├── main.ts
├── meta.ts
├── math/
│   ├── score.ts                       # analytical scores for known distributions
│   ├── losses.ts                      # ISM, DSM, sliced SM loss functions
│   ├── langevin.ts                    # Langevin sampler
│   ├── score-model.ts                 # MLP forward pass with (x, sigma) input
│   └── *.test.ts
├── sections/
│   ├── 01-hook.ts
│   ├── 02-score-function.ts
│   ├── 03-fisher-divergence.ts
│   ├── 04-implicit-score-matching.ts
│   ├── 05-denoising-score-matching.ts
│   ├── 06-langevin-dynamics.ts
│   ├── 07-annealed-langevin.ts
│   └── 08-where-youll-see-this.ts
├── viz/
│   ├── score-field-explorer.ts        # §2 — quiver plot for known distributions
│   ├── normalization-irrelevance.ts   # §2 — show s(x) unchanged when Z changes
│   ├── ism-derivation.ts              # §4 — animated integration by parts
│   ├── noise-smoothed-score.ts        # §5 — slider for sigma, watch field smooth
│   ├── dsm-target.ts                  # §5 — illustrate "predict -eps/sigma"
│   ├── langevin-sampler.ts            # §6 — particles flowing on analytic score
│   └── annealed-langevin.ts           # §7 — CENTERPIECE, learned score, annealed
├── assets/
│   └── score-weights.json             # pre-trained model
└── styles/
    └── overrides.css
```

---

## 2. Visual / Aesthetic Direction

Same paper-and-ink aesthetic. Lesson-local accents:

```css
--score-arrow:    #2c5f8d;   /* score-field arrows — slate blue */
--score-arrow-2:  #c87f3b;   /* secondary field (e.g., noisy version) — ochre */
--data-point:     #b8651a;   /* training-data points — burnt sienna */
--langevin-traj:  #5a8a6a;   /* particle trajectories — sage */
--noise-level:    #6b3a8c;   /* noise schedule indicators — plum */
--mode:           #d4a437;   /* density modes / target attractors — amber */
```

Two consistent visual conventions:
- **Score vectors** drawn as `--score-arrow` arrows with arrowheads,
  scaled so the largest vector in the field is ~5% of canvas width.
- **Particles undergoing Langevin** drawn as small dots with a
  fading trail in `--langevin-traj` showing recent positions.

---

## 3. Lesson Metadata (`src/lessons/score-matching/meta.ts`)

```ts
import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'score-matching',
  title: 'Score Matching',
  subtitle: 'Model the gradient of the log-density. Sample via Langevin.',
  tier: 3,
  difficulty: 3,
  estimatedHours: 4,
  status: 'planned',
  prerequisites: [
    { id: 'kl-jensen',         strength: 'required',    anchor: 'kl-definition' },
    { id: 'gaussian-cookbook', strength: 'required',    anchor: 'reparam-matrix' },
  ],
  recommendedNext: ['ddpm'],
  alsoUsedBy: ['ddpm'],
  description:
    'Score-based generative modeling: fit ∇ log p(x) directly, ' +
    'avoiding the normalization constant. Three tractable losses, ' +
    'Langevin sampling, and the noise-annealing trick that DDPM ' +
    'extends into a Markov chain.',
  exportedAnchors: {
    'score-definition':     'The score function: ∇_x log p(x)',
    'fisher-divergence':    'Fisher divergence and the score matching objective',
    'ism':                  'Implicit score matching (Hyvärinen integration by parts)',
    'dsm':                  'Denoising score matching (Vincent identity)',
    'langevin':             'Langevin dynamics for sampling from a known score',
    'annealed-langevin':    'Annealed Langevin sampling across noise levels',
  },
  path: '/lessons/score-matching',
};
```

---

## 4. Section-by-Section Plan

Eight sections, ~90-120 min of careful reading + interaction.

---

### Section 1 — Hook

**Length**: ~180 words.

**Prose** (verbatim):

> Every generative model so far has tried to evaluate or maximize
> $p(x)$, the data density. VAEs gave up on the density directly and
> went after a lower bound. EM iterated on the log-likelihood.
> Plain MLE just sets the gradient to zero.
>
> But there's another option, and it's a strange one: **don't model
> $p(x)$ at all. Model the gradient $\nabla_x \log p(x)$ instead.**
>
> Why would you do that? Two reasons. First, $\nabla_x \log p(x)$
> doesn't depend on the normalization constant. If $p(x) = \tilde{p}(x) / Z$,
> the $\log Z$ vanishes under the gradient. Models without
> normalization constants are radically more flexible — any function
> from $\mathbb{R}^d$ to $\mathbb{R}^d$ is a candidate. Second,
> there's a beautiful sampling algorithm — **Langevin dynamics** —
> that takes a gradient field and produces samples from the
> corresponding distribution.
>
> The two ideas together — model the score, then sample with Langevin
> — are called **score-based generative modeling**. This lesson sets
> them up. The next lesson, **DDPM**, weaves them into the largest
> family of generative models in the modern world.

CTA button: "Meet the score function →"

---

### Section 2 — The Score Function

**Length**: ~700 words. **Anchor: `score-definition`**.

**Prose**:

> The **score function** of a density $p$ is its gradient on a log
> scale:
>
> $$\boxed{\;\; s(x) \;:=\; \nabla_x \log p(x) \;\;}$$
>
> Beware: this name collides with the statistics convention where
> "score" means $\nabla_\theta \log p(x; \theta)$ — the gradient
> with respect to *parameters*. In score-based modeling, we always
> mean the gradient with respect to the *input* $x$, holding the
> distribution fixed. From now on, "score" means $\nabla_x \log p(x)$.
>
> #### Geometric meaning
>
> $s(x)$ is a vector field on $\mathbb{R}^d$. At every point $x$ it
> outputs a $d$-dimensional vector pointing in the direction of
> steepest ascent of $\log p$ — i.e., toward higher density. Three
> immediate observations:
>
> - **At modes** (local maxima of $p$), $s(x) = 0$. The score
>   vanishes wherever the density is locally peaked.
> - **Far from data**, where $p$ is small, the score points toward
>   regions where $p$ is larger. The score field "points home" no
>   matter where you are.
> - **Magnitude** of $\|s(x)\|$ measures how rapidly $\log p$ is
>   changing. Sharp peaks → large score nearby; broad plateaus →
>   small score.

> #### Why the normalization disappears
>
> Write any density as $p(x) = \tilde{p}(x) / Z$ where $\tilde{p}$ is
> any non-negative function and $Z = \int \tilde{p}(x) dx$ is the
> normalization constant. Then:
>
> $$\log p(x) \;=\; \log \tilde{p}(x) \;-\; \log Z$$
>
> Taking the gradient with respect to $x$:
>
> $$s(x) \;=\; \nabla_x \log \tilde{p}(x) \;-\; \nabla_x \log Z \;=\; \nabla_x \log \tilde{p}(x)$$
>
> because $\log Z$ doesn't depend on $x$. **The score is invariant
> to normalization.** This is the structural fact that makes
> score-based modeling work: we can specify an unnormalized energy
> $\tilde{p}(x) = e^{-U(x)}$ and the score is just $-\nabla U(x)$ —
> no integration over $\mathbb{R}^d$ required.

> #### Three worked examples
>
> **Standard normal.** If $p(x) = \mathcal{N}(x; 0, I)$, then
> $\log p(x) = -\tfrac{1}{2} \|x\|^2 + \text{const}$, so
>
> $$s(x) \;=\; -x$$
>
> A linear vector field pointing radially toward the origin. The
> "wind always blows you toward zero."
>
> **General Gaussian.** For $p(x) = \mathcal{N}(x; \mu, \Sigma)$:
>
> $$s(x) \;=\; -\Sigma^{-1}(x - \mu)$$
>
> Still linear, but now anisotropic: the score is rescaled and
> rotated by $\Sigma^{-1}$.
>
> At $\mu = 0$, $\Sigma = \begin{pmatrix}1 & 0.5 \\ 0.5 & 1\end{pmatrix}$,
> $x = (1, 0)$:
>
> $$s(x) \;=\; -\frac{1}{0.75}\begin{pmatrix}1 & -0.5 \\ -0.5 & 1\end{pmatrix}\begin{pmatrix}1 \\ 0\end{pmatrix} \;=\; \begin{pmatrix}-4/3 \\ 2/3\end{pmatrix}$$
>
> (Pre-verified; test target in `score.test.ts`.)
>
> **Gaussian mixture.** For $p(x) = \sum_k \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$:
>
> $$s(x) \;=\; \sum_k r_k(x) \cdot \big[-\Sigma_k^{-1}(x - \mu_k)\big]$$
>
> where $r_k(x) = \pi_k \mathcal{N}(x; \mu_k, \Sigma_k) / p(x)$ is the
> responsibility of component $k$ — same quantity as the E-step in
> [EM §4](../em/#q-function). The mixture score is the
> responsibility-weighted average of the component scores. **Saddles
> emerge** at points equidistant between modes, where competing
> components pull in opposite directions.

**Cross-link callout — back to EM** (`type=back`):

> **Uses: [EM §4 — the E-step](../em/#q-function)**. The
> responsibilities $r_k(x)$ that weight the mixture-score components
> are the same posterior-responsibility quantities that EM computes
> in its E-step. Different framework, same statistical object.

**Visualization 1 — `<ScoreFieldExplorer>`** (full width):

A 2D canvas showing the score field of a selectable distribution as
a quiver plot. Background: contours of $\log p$ in `--gauss-grid`
shading. Foreground: a grid of arrows at $20 \times 20$ sample
points, each arrow pointing along $s(x)$ scaled so the longest
arrow is ~5% of canvas width.

Dropdown selector for the distribution:
- Standard normal (linear field, points to origin)
- General Gaussian (user adjusts $\mu$ and $\Sigma$ via sliders)
- Gaussian mixture, 2 components (user adjusts the centers and
  mixing weights)
- Banana / Rosenbrock (curved field)
- Ring distribution (radial field with a hole)

Hover any arrow: tooltip shows $x$, $s(x)$, $\log p(x)$.

The point is to **build intuition** that "score = vector field" and
that different distributions have visually different score fields.

**Visualization 2 — `<NormalizationIrrelevance>`** (medium width):

A small, focused viz to drive home the "score doesn't depend on $Z$"
point. Two side-by-side canvases:
- Left: $p(x) = \frac{1}{Z}\exp(-U(x))$ with a slider for $Z$ (changes
  the **density** values but not the **shape**).
- Right: the score field, which is **literally unchanged** regardless
  of $Z$'s value.

User slides $Z$, watches left side rescale, watches right side stay
still. A bold annotation: "Score is invariant to normalization."

---

### Section 3 — Fisher Divergence and the Score Matching Objective

**Length**: ~500 words. **Anchor: `fisher-divergence`**.

**Prose**:

> To **fit** a score model $s_\theta(x)$ to data, we need a loss.
> The natural one: penalize the squared difference between the
> model's score and the data's true score.
>
> $$\boxed{\;\; \mathcal{L}_{\text{SM}}(\theta) \;=\; \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2\right] \;\;}$$
>
> This quantity — the expected squared $L^2$ distance between the
> two score fields — is the **Fisher divergence**
> $\mathcal{F}(p_{\text{data}} \,\|\, p_\theta)$, sometimes called the
> **relative Fisher information** in the literature.
>
> Properties (worth knowing, not derived here):
>
> - Non-negative: $\mathcal{F} \geq 0$, with equality iff
>   $s_\theta(x) = \nabla \log p_{\text{data}}(x)$ for $p_{\text{data}}$-almost-every $x$.
> - Asymmetric in $p_{\text{data}}$ and $p_\theta$ — like KL but
>   not the same quantity.
> - Connected to KL: De Bruijn's identity says
>   $\partial_t D_{\mathrm{KL}}(p * \mathcal{N}_t \,\|\, q * \mathcal{N}_t) = -\tfrac{1}{2}\mathcal{F}(p * \mathcal{N}_t, q * \mathcal{N}_t)$
>   where $p * \mathcal{N}_t$ is $p$ convolved with Gaussian noise of
>   variance $t I$. Fisher divergence is KL's "derivative through
>   noise smoothing." A deep result; we won't use it but it's why
>   this all works.

> #### The fundamental problem
>
> The loss $\mathcal{L}_{\text{SM}}$ requires
> $\nabla_x \log p_{\text{data}}(x)$ — the score of the data
> distribution — which we don't have. We only have samples from
> $p_{\text{data}}$, not its score field.
>
> Three workarounds, each removing the dependence on
> $\nabla \log p_{\text{data}}$ in a different way:
>
> 1. **Implicit score matching** (Hyvärinen 2005) — integration by
>    parts. Trades $\nabla \log p_{\text{data}}$ for a trace of the
>    Jacobian of $s_\theta$. **§4.**
> 2. **Denoising score matching** (Vincent 2011) — corrupt the data
>    with noise and match the score of the noise-perturbed
>    distribution, which has a tractable conditional. **§5.**
> 3. **Sliced score matching** (Song et al. 2019) — project onto
>    random directions, avoid the full Jacobian. Mentioned briefly;
>    not the focus.
>
> §4 develops route 1. §5 develops route 2 — the one that DDPM uses.

**Visualization 3 — Conceptual diagram** (small, illustrative):

A simple diagram showing $\mathcal{L}_{\text{SM}}$ as the gap between
two vector fields (data score in `--score-arrow`, model score in
`--score-arrow-2`). Five sample points; at each point, the squared
distance between the two arrows is highlighted. A label "this is the
quantity we want to minimize — but we don't have access to the data
score."

The visualization is not interactive — just a setup for understanding
why the two tractable routes are needed.

---

### Section 4 — Implicit Score Matching (Hyvärinen)

**Length**: ~700 words. **Anchor: `ism`**. The conceptually-hardest
section.

**Prose**:

> The first route to a tractable score-matching loss is to **rewrite
> the inaccessible term using integration by parts.** This was
> Hyvärinen's original (2005) contribution.
>
> Expand the squared norm in $\mathcal{L}_{\text{SM}}$:
>
> $$\mathcal{L}_{\text{SM}} \;=\; \mathbb{E}_{p_{\text{data}}}\!\left[\|s_\theta(x)\|^2\right] \;-\; 2\, \mathbb{E}_{p_{\text{data}}}\!\left[s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\right] \;+\; \mathbb{E}_{p_{\text{data}}}\!\left[\|\nabla_x \log p_{\text{data}}(x)\|^2\right]$$
>
> The third term doesn't depend on $\theta$ — drop it. The first
> term is easy: just sample $x \sim p_{\text{data}}$, evaluate
> $\|s_\theta(x)\|^2$, average. The middle term is the problem; let
> me massage it.

> #### The integration-by-parts trick
>
> Use the identity $p_{\text{data}}(x) \nabla \log p_{\text{data}}(x) = \nabla p_{\text{data}}(x)$:
>
> $$\mathbb{E}_{p_{\text{data}}}\!\left[s_\theta(x)^\top \nabla \log p_{\text{data}}(x)\right] \;=\; \int p_{\text{data}}(x) \, s_\theta(x)^\top \nabla \log p_{\text{data}}(x) \, dx \;=\; \int s_\theta(x)^\top \nabla p_{\text{data}}(x) \, dx$$
>
> Now integrate by parts in each coordinate. For coordinate $i$:
>
> $$\int s_{\theta,i}(x) \, \frac{\partial p_{\text{data}}}{\partial x_i} \, dx \;=\; \underbrace{\left[s_{\theta,i}(x) \, p_{\text{data}}(x)\right]_{-\infty}^{\infty}}_{= 0} \;-\; \int p_{\text{data}}(x) \, \frac{\partial s_{\theta,i}}{\partial x_i} \, dx$$
>
> The boundary term vanishes **assuming** $p_{\text{data}}(x) \to 0$
> as $\|x\| \to \infty$ (true for any reasonable data distribution)
> and $s_\theta(x)$ doesn't grow too fast (true for any neural
> network with bounded weights). Summing over $i$:
>
> $$\mathbb{E}_{p_{\text{data}}}\!\left[s_\theta(x)^\top \nabla \log p_{\text{data}}(x)\right] \;=\; -\mathbb{E}_{p_{\text{data}}}\!\left[\sum_i \frac{\partial s_{\theta,i}(x)}{\partial x_i}\right] \;=\; -\mathbb{E}_{p_{\text{data}}}\!\left[\mathrm{tr}\!\left(\nabla s_\theta(x)\right)\right]$$
>
> The trace of the Jacobian of $s_\theta$. **No more
> $\nabla \log p_{\text{data}}$.** Substituting back:
>
> $$\boxed{\;\; \mathcal{L}_{\text{ISM}}(\theta) \;=\; \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\|s_\theta(x)\|^2 \;+\; 2\,\mathrm{tr}\!\left(\nabla s_\theta(x)\right)\right] \;\;}$$
>
> Up to the constant we dropped, this **equals** $\mathcal{L}_{\text{SM}}$,
> but every term is computable from data samples alone. This is the
> implicit score matching loss.

> #### Sanity check
>
> Let $p_{\text{data}} = \mathcal{N}(0, I_d)$ and parameterize
> $s_\theta(x) = a x$ for a single scalar $a$. Then
> $\|s_\theta\|^2 = a^2 \|x\|^2$ and
> $\mathrm{tr}(\nabla s_\theta) = a d$. So:
>
> $$\mathcal{L}_{\text{ISM}} \;=\; a^2 \, \mathbb{E}\|x\|^2 \;+\; 2 a d \;=\; a^2 \cdot d \;+\; 2 a d \;=\; d(a^2 + 2a)$$
>
> Minimizing over $a$: derivative is $d(2a + 2) = 0$, so $a = -1$.
> This recovers $s_\theta(x) = -x$, the **exact** score of $\mathcal{N}(0, I_d)$
> (from §2). The framework works.

> #### The catch: trace cost
>
> Computing $\mathrm{tr}(\nabla s_\theta(x))$ exactly requires $d$
> backward passes (one per coordinate) or one Hessian-vector product
> per coordinate. For image data with $d = 10^5$ or more, this is
> prohibitive.
>
> Two ways around it:
>
> - **Sliced score matching** (Song et al. 2019): replace the trace
>   with a Hutchinson trace estimator using random projections.
>   $O(1)$ extra cost per step.
> - **Denoising score matching** (Vincent 2011): avoid the trace
>   entirely by changing what we're matching against. **§5.**
>
> The DSM route is what generative diffusion models use. It's where
> we go next.

**Cross-link callout — back to Gaussian Cookbook** (`type=back`):

> **Uses: [Gaussian Cookbook §2 — score of a Gaussian](../gaussian-cookbook/#mvn-density)**.
> The sanity-check example above uses the fact that the score of
> $\mathcal{N}(0, I)$ is $s(x) = -x$ — a direct corollary of the
> Cookbook's affine-transformation rule for Gaussian densities.

**Visualization 4 — `<ISMDerivation>`** (medium width):

An animated step-through of the derivation above. The user clicks
through five panels:
1. Start: $\mathcal{L}_{\text{SM}} = \mathbb{E}[\|s_\theta - s_{\text{data}}\|^2]$.
2. Expand the square.
3. Drop the $\theta$-independent term.
4. Apply integration by parts on the cross term.
5. Arrive at $\mathcal{L}_{\text{ISM}} = \mathbb{E}[\|s_\theta\|^2 + 2 \mathrm{tr}(\nabla s_\theta)]$.

Each panel highlights the term that changes; previous terms are
greyed. A "play all" button animates through automatically. This is
mostly a static visual; the value is the rhythmic step-by-step.

---

### Section 5 — Denoising Score Matching (Vincent)

**Length**: ~900 words. **Anchor: `dsm`**. The centerpiece equation.

**Prose**:

> Vincent (2011) found a different way around the
> $\nabla \log p_{\text{data}}$ problem: **don't match the data
> score; match the score of a noise-corrupted version.**
>
> #### Setup
>
> Add Gaussian noise to data points:
>
> $$\tilde{x} \;=\; x + \sigma \varepsilon, \qquad x \sim p_{\text{data}}, \quad \varepsilon \sim \mathcal{N}(0, I)$$
>
> The noisy variable $\tilde{x}$ follows the **noise-perturbed
> distribution** $p_\sigma$:
>
> $$p_\sigma(\tilde{x}) \;=\; \int p_{\text{data}}(x) \, \mathcal{N}(\tilde{x}; x, \sigma^2 I) \, dx$$
>
> $p_\sigma$ is $p_{\text{data}}$ convolved with a Gaussian kernel
> of width $\sigma$. It's **smoother** than $p_{\text{data}}$, and as
> $\sigma \to 0$ it converges to $p_{\text{data}}$. As $\sigma \to \infty$
> it converges to the kernel — a wide Gaussian centered at the data
> mean.
>
> Our model $s_\theta(\tilde{x}, \sigma)$ now takes both the noisy
> input **and** the noise level $\sigma$, and tries to match the
> score of $p_\sigma$:
>
> $$\mathcal{L}_{\sigma}(\theta) \;=\; \mathbb{E}_{\tilde{x} \sim p_\sigma}\!\left[\|s_\theta(\tilde{x}, \sigma) - \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})\|^2\right]$$
>
> Same Fisher-divergence shape as before, just for the smoothed
> distribution. We still don't know $\nabla \log p_\sigma$ — that
> looks like the same problem.
>
> But Vincent's beautiful identity rescues us.

> #### Vincent's identity
>
> $$\boxed{\;\; \mathbb{E}_{\tilde{x} \sim p_\sigma}\!\left[\|s_\theta(\tilde{x}, \sigma) - \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})\|^2\right] \;=\; \mathbb{E}_{x \sim p_{\text{data}}, \, \varepsilon \sim \mathcal{N}(0, I)}\!\left[\|s_\theta(\tilde{x}, \sigma) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x)\|^2\right] \;+\; C \;\;}$$
>
> where $C$ is a constant in $\theta$ and
> $q_\sigma(\tilde{x} \mid x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$
> is the **conditional** Gaussian (clean $x$ → noisy $\tilde{x}$).
>
> The right side is tractable. The conditional score is just the
> score of a Gaussian:
>
> $$\nabla_{\tilde{x}} \log \mathcal{N}(\tilde{x}; x, \sigma^2 I) \;=\; -\frac{\tilde{x} - x}{\sigma^2} \;=\; -\frac{\varepsilon}{\sigma}$$
>
> Substituting:
>
> $$\boxed{\;\; \mathcal{L}_{\text{DSM}}(\theta; \sigma) \;=\; \mathbb{E}_{x, \varepsilon}\!\left[\,\left\|s_\theta(x + \sigma \varepsilon, \sigma) \;+\; \frac{\varepsilon}{\sigma}\right\|^2\right] \;\;}$$
>
> **The model takes a noisy input $\tilde{x} = x + \sigma \varepsilon$
> and learns to predict $-\varepsilon / \sigma$.** No Jacobian. No
> trace. Just MSE between two vectors.
>
> #### Why this works (proof sketch, in `<ProofToggle>`)
>
> Expand the left side, focusing on the cross term:
>
> $$-2 \, \mathbb{E}_{p_\sigma}[s_\theta(\tilde{x}, \sigma)^\top \nabla \log p_\sigma(\tilde{x})]$$
>
> Write $p_\sigma(\tilde{x}) = \int p_{\text{data}}(x) q_\sigma(\tilde{x} \mid x) dx$
> and apply the same identity used in §4:
> $p_\sigma(\tilde{x}) \nabla \log p_\sigma(\tilde{x}) = \nabla p_\sigma(\tilde{x})$.
> Then:
>
> $$\mathbb{E}_{p_\sigma}[s_\theta^\top \nabla \log p_\sigma] \;=\; \int s_\theta(\tilde{x}, \sigma)^\top \nabla p_\sigma(\tilde{x}) d\tilde{x}$$
>
> Substitute $p_\sigma(\tilde{x}) = \int p_{\text{data}}(x) q_\sigma(\tilde{x} \mid x) dx$
> and move the gradient inside:
>
> $$= \int s_\theta(\tilde{x}, \sigma)^\top \int p_{\text{data}}(x) \nabla q_\sigma(\tilde{x} \mid x) \, dx \, d\tilde{x}$$
>
> $$= \int \int p_{\text{data}}(x) q_\sigma(\tilde{x} \mid x) s_\theta(\tilde{x}, \sigma)^\top \nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x) \, dx \, d\tilde{x}$$
>
> $$= \mathbb{E}_{x \sim p_{\text{data}}, \, \tilde{x} \sim q_\sigma(\cdot \mid x)}\!\left[s_\theta(\tilde{x}, \sigma)^\top \nabla \log q_\sigma(\tilde{x} \mid x)\right]$$
>
> So the cross term in $\mathcal{L}_\sigma$ — which originally had
> $\nabla \log p_\sigma$ — equals the cross term with
> $\nabla \log q_\sigma(\tilde{x} \mid x)$ instead. The other terms
> match too (with the constant $C$ absorbing the
> $\|\nabla \log p_\sigma\|^2 - \|\nabla \log q_\sigma\|^2$
> discrepancy, which doesn't depend on $\theta$). $\blacksquare$

> #### What the model learns to predict
>
> Different equivalent parameterizations:
>
> - **Score parameterization**: $s_\theta(\tilde{x}, \sigma)$ directly
>   outputs an estimate of $\nabla \log p_\sigma(\tilde{x})$. Target:
>   $-\varepsilon / \sigma$.
> - **Noise parameterization** ($\varepsilon$-prediction): the network
>   $\varepsilon_\theta(\tilde{x}, \sigma)$ outputs an estimate of
>   $\varepsilon$. Target: $\varepsilon$. Score recovered as
>   $s_\theta = -\varepsilon_\theta / \sigma$. **DDPM uses this
>   parameterization** — the network is asked "what noise was added?",
>   not "what's the score?".
> - **Clean-data parameterization** ($x$-prediction): the network
>   $x_\theta(\tilde{x}, \sigma)$ outputs an estimate of $x$. Target: $x$.
>   Score recovered as $s_\theta = (x_\theta - \tilde{x}) / \sigma^2$.
>   Used by some diffusion variants.
>
> All three are mathematically equivalent up to a per-$\sigma$
> rescaling of the loss. They differ in optimization dynamics and
> numerical behavior. **For DDPM, $\varepsilon$-prediction with a
> per-timestep weighting of 1 was empirically best.**

> #### Worked numerical example
>
> Take a clean data point $x = (0.5, 1.0)$, a noise level $\sigma = 0.1$,
> and a noise sample $\varepsilon = (0.3, -0.5)$. Then:
>
> - Noisy input: $\tilde{x} = x + \sigma \varepsilon = (0.53, 0.95)$.
> - Score target: $-\varepsilon / \sigma = (-3, 5)$.
> - Equivalent $\varepsilon$-prediction target: $(0.3, -0.5)$.
> - Loss at a randomly-initialized model output $s_\theta = (0, 0)$:
>   $\|(0, 0) - (-3, 5)\|^2 = 9 + 25 = 34$.
>
> (Pre-verified; test target in `losses.test.ts`.)

**Cross-link callout — back to Gaussian Cookbook** (`type=back`):

> **Uses: [Gaussian Cookbook §4 — reparameterization](../gaussian-cookbook/#reparam-matrix)**.
> The construction $\tilde{x} = x + \sigma \varepsilon$ is the
> reparameterization trick: a deterministic transform of fixed noise
> $\varepsilon$, with $x$ acting as the "mean" and $\sigma$ as the
> scale. This is what makes the DSM loss differentiable in any
> downstream pipeline that needs $\partial \mathcal{L} / \partial x$.

**Visualization 5 — `<NoiseSmoothedScore>`** (full width):

A 2D canvas showing the analytical score field of a Gaussian mixture
(two modes at $(\pm 2, 0)$, each $\Sigma = 0.2 I$). A slider for
$\sigma$ from 0.01 to 3.0 controls the noise level. As $\sigma$
increases, the canvas re-renders the score field of $p_\sigma$
(computable in closed form for Gaussian mixtures — see math module).

What the user sees:
- **Small $\sigma$** (near 0): the score field is sharp near the
  modes and chaotic between them. Far from the data manifold, the
  field is essentially zero (no signal there).
- **Medium $\sigma$**: the field smooths out. There's now signal
  everywhere on the canvas — even points far from the data have a
  score pointing "toward" the smoothed density.
- **Large $\sigma$**: the field collapses to the score of a single
  wide Gaussian centered at the data mean. The mixture structure
  has been washed out.

The slider has annotated stops at "data noise level" (small),
"diffusion training" (medium), "prior noise" (large).

Pedagogical point: **single-noise-level score matching is brittle.
Multi-noise-level training (NCSN, DDPM) gives the model signal at
every scale.** This sets up §7.

**Visualization 6 — `<DSMTarget>`** (medium width):

A focused illustration. A 2D canvas with a single data point $x$ (in
`--data-point`). The user clicks anywhere on the canvas; that's
the noisy $\tilde{x}$. The vector from $\tilde{x}$ to $x$ is drawn,
and the **DSM target** $-(\tilde{x} - x)/\sigma^2$ is drawn as a
scaled arrow at $\tilde{x}$. As the user clicks farther from $x$, the
target arrow points back toward $x$ more strongly (because the
displacement is larger). The annotation: "the model learns to predict
the direction back to plausible data, scaled by noise variance."

---

### Section 6 — Langevin Dynamics for Sampling

**Length**: ~700 words. **Anchor: `langevin`**.

**Prose**:

> We now have a trained score model $s_\theta(x) \approx \nabla \log p(x)$.
> The question: how do we use it to **sample** from $p$?
>
> The answer is **Langevin dynamics**, a stochastic process that uses
> the score as a drift term and Gaussian noise to explore.

> #### The continuous-time form
>
> Langevin dynamics is the SDE
>
> $$\boxed{\;\; dx_t \;=\; \nabla_x \log p(x_t) \, dt \;+\; \sqrt{2} \, dW_t \;\;}$$
>
> where $W_t$ is a standard Brownian motion. Under mild regularity
> conditions, **the stationary distribution of this SDE is exactly $p$**
> — start any trajectory anywhere, let it run long enough, and the
> resulting distribution converges to $p$.
>
> Two competing forces:
> - **Drift**: $\nabla \log p$ pulls $x_t$ toward higher density.
>   Without noise, this would be gradient ascent — converging to a
>   mode.
> - **Diffusion**: $\sqrt{2} dW_t$ adds Gaussian noise. Prevents
>   collapse to modes; encourages exploration.
>
> The balance is set so that detailed balance holds with respect to
> $p$ — the Fokker-Planck equation has $p$ as stationary solution.

> #### The discrete-time algorithm
>
> Discretize the SDE with step size $\eta$:
>
> $$\boxed{\;\; x_{t+1} \;=\; x_t \;+\; \eta \, \nabla \log p(x_t) \;+\; \sqrt{2 \eta} \, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, I) \;\;}$$
>
> Run this for many steps; the empirical distribution of $x_t$
> approaches $p$. For small $\eta$ and large $t$, the approximation
> is good. For large $\eta$, there's discretization bias.
>
> **In practice we replace $\nabla \log p$ with $s_\theta$**:
>
> $$x_{t+1} \;=\; x_t \;+\; \eta \, s_\theta(x_t) \;+\; \sqrt{2 \eta} \, \varepsilon_t$$
>
> This is **score-based Langevin sampling**.

> #### Worked example: one step on $\mathcal{N}(0, I)$
>
> Start at $x_0 = (1, 1)$. The true score (we'll use the analytical
> one for illustration) is $s(x_0) = -x_0 = (-1, -1)$. Take step
> size $\eta = 0.1$ and noise sample $\varepsilon = (0.2, -0.3)$.
>
> $$x_1 \;=\; (1, 1) + 0.1 \cdot (-1, -1) + \sqrt{0.2} \cdot (0.2, -0.3) \;=\; (1, 1) + (-0.1, -0.1) + (0.0894, -0.1342)$$
>
> $$x_1 \;\approx\; (0.9894, 0.7658)$$
>
> The drift pulled $x$ toward the origin; the noise nudged it
> slightly off-axis. (Pre-verified; test target in `langevin.test.ts`.)

> #### Why naive Langevin fails on real data
>
> Plain Langevin with a learned $s_\theta$ has two failure modes:
>
> 1. **The score is unreliable off the data manifold.** Real data
>    lives on a low-dimensional manifold inside high-dim space. The
>    score model is trained on $p_{\text{data}}$ — it never sees
>    points far from the manifold, so $s_\theta$ there is meaningless.
>    A particle initialized in this no-signal region wanders
>    randomly.
> 2. **Multimodal mixing is slow.** Even when $s_\theta$ is accurate,
>    crossing between modes of a multimodal $p$ requires the noise
>    to randomly drive $x_t$ over a low-density barrier. With small
>    $\eta$, this can take exponentially many steps.
>
> Both problems point in the same direction: **train the score at
> multiple noise levels.** At high noise, the score field is smooth
> and globally informative; at low noise, the score field is sharp
> and locally precise. **Annealed Langevin sampling** uses both —
> §7.

**Visualization 7 — `<LangevinSampler>`** (full width):

A 2D canvas with a fixed target distribution (selectable from a small
menu: 2D Gaussian, 2-mode mixture, ring distribution). The score is
analytical (not learned) — we want the reader to see Langevin work
when the score is correct.

User controls:
- Number of particles (1, 10, 100).
- Step size $\eta$.
- "Step", "Play", "Pause", "Reset" buttons.

What happens:
- Particles start from $\mathcal{N}(0, 9 I)$ (broad initialization).
- Each click of "Step" applies one Langevin update.
- "Play" animates continuous updates.
- Trajectories fade in `--langevin-traj`; particles in `--data-point`.

After a few hundred steps with the 2D Gaussian, particles cluster at
the origin. With a mixture, they cluster at both modes — but the
**split between modes depends on initialization** and the small-$\eta$
limit takes many steps to equilibrate. The user can experimentally
discover the mixing-time problem this section mentions.

---

### Section 7 — Annealed Langevin and the Path to DDPM

**Length**: ~700 words. **Anchor: `annealed-langevin`**. The
centerpiece.

**Prose**:

> The fix to Langevin's failure modes is **anneal the noise level**:
> train and sample at a sequence of decreasing $\sigma$ values.
> Song & Ermon (2019) introduced this as **Noise Conditional Score
> Networks (NCSN)**. It's the immediate predecessor of DDPM.
>
> #### Training: score at many noise levels
>
> Train $s_\theta(x, \sigma)$ such that, for every $\sigma$ in a
> chosen schedule $\{\sigma_1 > \sigma_2 > \cdots > \sigma_L\}$,
>
> $$s_\theta(x, \sigma_\ell) \;\approx\; \nabla_x \log p_{\sigma_\ell}(x)$$
>
> The training loss is a weighted sum of DSM losses, one per noise
> level:
>
> $$\mathcal{L}(\theta) \;=\; \sum_{\ell=1}^{L} \lambda(\sigma_\ell) \cdot \mathbb{E}_{x, \varepsilon}\!\left[\,\left\|s_\theta(x + \sigma_\ell \varepsilon, \sigma_\ell) + \frac{\varepsilon}{\sigma_\ell}\right\|^2\right]$$
>
> The weighting $\lambda(\sigma)$ is typically chosen so each term
> has comparable scale: $\lambda(\sigma) = \sigma^2$ rescales the
> target to be $-\sigma \varepsilon$ (variance ~1) and the prediction
> to be $\sigma \cdot s_\theta$.
>
> One **network** outputs scores at all noise levels — typically by
> conditioning on $\sigma$ (or $\log \sigma$, or an embedding of it).

> #### Sampling: annealed Langevin
>
> Run Langevin from the largest noise level down:
>
> ```
> # Initialize at the largest noise level (essentially noise)
> x ~ N(0, sigma_1^2 * I)
>
> for ell = 1 to L:
>     # Step size scales with current noise level (NCSN's recipe)
>     alpha_ell = epsilon * (sigma_ell / sigma_L)^2
>
>     for t = 1 to T:
>         eps_t ~ N(0, I)
>         x = x + (alpha_ell / 2) * s_theta(x, sigma_ell)
>               + sqrt(alpha_ell) * eps_t
>
> # x at the end is a sample from (approximately) p_sigma_L,
> # which is close to p_data when sigma_L is small
> return x
> ```
>
> Each "outer" iteration uses the score at the current noise level
> for $T$ steps of Langevin, then decreases the noise level. The
> intuition:
>
> - **Early (large $\sigma$)**: the score field is smooth, defined
>   everywhere on the canvas, points roughly toward the data mean.
>   This phase rapidly transports particles from initialization
>   toward the rough vicinity of the data.
> - **Middle**: the noise level shrinks; the score field develops
>   structure; particles localize toward modes.
> - **Late (small $\sigma$)**: the score field is sharp; particles
>   refine to high precision near the data manifold.
>
> Each phase **uses the score at exactly the noise level the
> particle is currently swimming in**. Mismatches (using a
> low-$\sigma$ score on a high-noise sample) cause artifacts; the
> noise-conditioning makes the model robust.

> #### NCSN, DDPM, and the modern landscape
>
> Three closely-related formulations:
>
> - **NCSN (Song & Ermon 2019)**: train one $s_\theta(x, \sigma)$
>   at $L$ discrete noise levels; sample by annealed Langevin.
> - **DDPM (Ho et al. 2020)**: train one $\varepsilon_\theta(x_t, t)$
>   at $T$ discrete timesteps (typically $T = 1000$). The forward
>   process is a Markov chain with a specific structure; the
>   sampling process is the reverse Markov chain, which turns out
>   to be Langevin-like at each step. The training loss simplifies
>   to a re-weighted denoising score matching loss. **Next lesson.**
> - **Score-based SDEs (Song et al. 2021)**: take the continuous-time
>   limit; the noise schedule becomes a continuous SDE, and sampling
>   is solving the reverse-time SDE. Unifies NCSN and DDPM.
>
> The differences are in parameterization, loss weighting, and
> discretization — not in the underlying idea. **The idea is what §5
> established and §7 generalizes: model the score at many noise
> levels, sample by walking down the noise schedule.**

**Visualization 8 — `<AnnealedLangevin>`** (full width, ≥720px tall).
**The centerpiece.** Layout:

```
┌──────────────────────────────────────────────────────────────────┐
│  DATA SPACE (x ∈ R²) — animation canvas                          │
│                                                                  │
│  • Training data (background, faded, 4-cluster mixture)          │
│  • Active particles (n = 100, in --langevin-traj)                │
│  • Particle trails (fading)                                      │
│  • Current score field overlay (toggleable, in --score-arrow)    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│  NOISE SCHEDULE                                                  │
│  σ_1 ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─● σ_L            │
│       ▲ current level                                            │
│                                                                  │
│  CONTROLS                                                        │
│  [Play / Pause]  [Step]  [Reset]                                 │
│  Toggle: ☑ show score field  ☐ show particle trails              │
│  Schedule: ◯ short (10 levels)  ◉ standard (50)  ◯ long (200)    │
└──────────────────────────────────────────────────────────────────┘
```

Behavior:
- On load: particles are initialized at $\mathcal{N}(0, \sigma_1^2 I)$ —
  a wide cloud essentially uniform across the canvas.
- The user clicks "Play". Particles step under Langevin at the
  current $\sigma$; after $T$ inner steps, the noise level decreases.
  This continues all the way down the schedule.
- By the end: particles cluster at the four data modes. The
  distribution of final positions visually matches the training
  data.
- The "score field overlay" toggle: when on, draw the learned score
  field at the **current** $\sigma$ as faint arrows. Watch the field
  morph from "smooth and globally directional" at large $\sigma$ to
  "sharp and modal" at small $\sigma$. **This is the visual payoff
  of the entire lesson.**

The trained score model required: see §5 of this spec for the
training notebook. Same architecture as the §8 viz of the VAE
lesson — a tiny MLP, weights as JSON.

**Cross-link callout — forward to DDPM** (`type=forward`):

> **Comes back in: [DDPM](../ddpm/)**. The DDPM training loss is
> exactly this — denoising score matching at multiple noise levels —
> with a specific parameterization (predict $\varepsilon$, not the
> score) and a specific noise schedule (the $\bar\alpha_t$ schedule).
> The DDPM sampling process is annealed Langevin, derived from a
> Markov-chain perspective rather than a noise-schedule perspective.

---

### Section 8 — Where You'll See This

**Length**: ~250 words.

**Prose**:

> Score-based modeling is the conceptual scaffolding for nearly all
> modern generative models.
>
> #### Coming next
>
> - **DDPM** — the canonical diffusion paper. The training loss
>   reduces to denoising score matching at $T$ timesteps. The
>   sampling process is reverse-time annealed Langevin.
>
> #### Adjacent / sidebar
>
> - **Score-based SDEs** (Song et al. 2021) — continuous-time view.
>   Unifies NCSN and DDPM as discretizations of a single SDE.
> - **EDM** (Karras et al. 2022) — refined noise schedules,
>   $\sigma$-conditioning, and a unified parameterization. Modern
>   diffusion foundation.
> - **Flow matching** (Lipman et al. 2023) — bridges score matching
>   and normalizing flows. Trains a vector field, not a score, but
>   the mathematical machinery is parallel.
> - **Energy-based models** — the framework where score matching
>   originated. Models specify $\tilde{p}(x) = e^{-U(x)}$ directly,
>   and score matching avoids estimating $Z$.
> - **Langevin sampling in molecular dynamics** — the same SDE is
>   used in physical simulations of molecular motion.
>
> #### Incoming references
>
> - **KL & Jensen** — Fisher divergence (§3) is connected to KL via
>   De Bruijn's identity (mentioned, not derived).
> - **Gaussian Cookbook** — the score of a Gaussian (§2), the
>   reparameterization construction (§5), and the noise-smoothed
>   density all use Cookbook identities.

**Visualization 9 — `<RoadmapMini>`** highlighting the lesson and
showing the two-branch convergence into DDPM (Score Matching from
KL + Cookbook; VAE from ELBO + Cookbook).

---

## 5. Algorithm / Math Implementation

### `src/lessons/score-matching/math/score.ts`

```ts
import { Matrix, inverse } from 'ml-matrix';

/** Score of a single multivariate Gaussian. */
export function scoreGaussian(x: number[], mu: number[], Sigma: number[][]): number[] {
  const SinvM = inverse(new Matrix(Sigma));
  const diffV = Matrix.columnVector(x.map((xi, i) => xi - mu[i]));
  return SinvM.mmul(diffV).to1DArray().map(v => -v);
}

/** Score of a Gaussian mixture: weighted sum of component scores. */
export function scoreGMM(
  x: number[],
  pis: number[],
  mus: number[][],
  Sigmas: number[][][],
): number[] {
  const d = x.length;
  // Compute log-probs of each component (for stable responsibilities)
  const logProbs = mus.map((mu, k) => logMVN(x, mu, Sigmas[k]) + Math.log(pis[k]));
  const m = Math.max(...logProbs);
  const norm = m + Math.log(logProbs.reduce((s, lp) => s + Math.exp(lp - m), 0));
  const rs = logProbs.map(lp => Math.exp(lp - norm));

  const s = new Array(d).fill(0);
  for (let k = 0; k < mus.length; k++) {
    const sk = scoreGaussian(x, mus[k], Sigmas[k]);
    for (let i = 0; i < d; i++) s[i] += rs[k] * sk[i];
  }
  return s;
}

/** Score of the noise-smoothed version p_sigma of a Gaussian mixture.
 *  Closed-form because convolution of a GMM with N(0, sigma^2 I) is a GMM. */
export function scoreSmoothedGMM(
  x: number[],
  pis: number[],
  mus: number[][],
  Sigmas: number[][][],
  sigma: number,
): number[] {
  const d = x.length;
  const Id = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => (i === j ? sigma * sigma : 0))
  );
  const smoothedSigmas = Sigmas.map(S =>
    S.map((row, i) => row.map((v, j) => v + Id[i][j]))
  );
  return scoreGMM(x, pis, mus, smoothedSigmas);
}

function logMVN(x: number[], mu: number[], Sigma: number[][]): number {
  // (see Cookbook math/mvn.ts; ported for self-containment of this test)
  // ...
}
```

### `src/lessons/score-matching/math/losses.ts`

```ts
/** Denoising score matching target: -(x_noisy - x_clean) / sigma^2 = -eps/sigma. */
export function dsmTarget(x_clean: number[], x_noisy: number[], sigma: number): number[] {
  return x_clean.map((c, i) => -(x_noisy[i] - c) / (sigma * sigma));
}

/** DSM loss for one (x, eps, sigma, model-prediction) tuple. */
export function dsmLoss(
  s_theta: number[],   // model output at noisy input
  x_clean: number[],
  eps: number[],
  sigma: number,
): number {
  let s = 0;
  for (let i = 0; i < x_clean.length; i++) {
    const target = -eps[i] / sigma;
    s += (s_theta[i] - target) ** 2;
  }
  return s;
}
```

### `src/lessons/score-matching/math/langevin.ts`

```ts
/** One step of Langevin dynamics: x' = x + eta * score(x) + sqrt(2*eta) * eps. */
export function langevinStep(
  x: number[],
  score: number[],
  eta: number,
  eps: number[],
): number[] {
  const sqrt2eta = Math.sqrt(2 * eta);
  return x.map((xi, i) => xi + eta * score[i] + sqrt2eta * eps[i]);
}

/** Annealed Langevin: walk down a noise schedule, T steps per level. */
export function annealedLangevin(
  x_init: number[],
  sigmas: number[],
  T: number,
  scoreFn: (x: number[], sigma: number) => number[],
  epsilonBase: number,
  rngNormal: () => number[],
): number[] {
  let x = x_init.slice();
  const sigma_L = sigmas[sigmas.length - 1];
  for (const sigma of sigmas) {
    const alpha = epsilonBase * (sigma / sigma_L) ** 2;
    for (let t = 0; t < T; t++) {
      const s = scoreFn(x, sigma);
      const eps = rngNormal();
      x = langevinStep(x, s, alpha / 2, eps);
    }
  }
  return x;
}
```

### Test cases

- `scoreGaussian((1, 0), (0, 0), [[1, 0.5], [0.5, 1]])` ≈ $(-4/3, 2/3)$.
- `dsmTarget((0.5, 1.0), (0.53, 0.95), 0.1)` ≈ $(-3, 5)$.
- `dsmLoss((0, 0), (0.5, 1.0), (0.3, -0.5), 0.1)` = $9 + 25 = 34$.
- `langevinStep((1, 1), (-1, -1), 0.1, (0.2, -0.3))` ≈ $(0.9894, 0.7658)$.
- ISM sanity check: numerically minimize `ism_loss(a) = d * (a^2 + 2a)`
  in `a` for d=2; recover $a = -1$.

---

## 6. Component Catalog

### Shared (already exist)
Standard chrome from `@shared/ui/`.

### Lesson-local
- `<ScoreFieldExplorer>` (§2)
- `<NormalizationIrrelevance>` (§2)
- `<ISMDerivation>` (§4)
- `<NoiseSmoothedScore>` (§5)
- `<DSMTarget>` (§5)
- `<LangevinSampler>` (§6)
- `<AnnealedLangevin>` (§7) — **the centerpiece**

---

## 7. Page-Level UX

Same as other lessons. `<PrereqStrip>` shows two prereqs: KL & Jensen
and Gaussian Cookbook (both required). One note specific to this
lesson:

1. **The §7 centerpiece is the polish budget sink** — invest most
   visual polish there. Smooth animation across noise levels matters
   more than perfect mathematical accuracy of the trained model.
2. **The §4 ISM derivation animation is a close second**. The
   integration-by-parts step is the conceptually-hardest moment of
   the lesson; the animation has to make each step land.

---

## 8. Acceptance Criteria

A learner who has worked through this page should be able to, on a
blank sheet:

1. Define the score function and explain why it doesn't depend on
   the normalization constant.
2. State the score of $\mathcal{N}(\mu, \Sigma)$ from memory.
3. Recall the implicit score matching loss
   $\mathcal{L}_{\text{ISM}} = \mathbb{E}[\|s_\theta\|^2 + 2 \mathrm{tr}(\nabla s_\theta)]$
   and outline the integration-by-parts derivation.
4. State the denoising score matching loss and identify
   $-\varepsilon / \sigma$ as the target.
5. Write the Langevin update equation and explain why both the
   drift and the noise are necessary.
6. Explain in two sentences why single-noise-level Langevin sampling
   fails on real data and how annealing fixes it.
7. Recognize that the DDPM training loss is denoising score matching
   at $T$ noise levels.

---

## 9. Stretch Goals (post-MVP)

- **Sliced score matching** (Song et al. 2019) — Hutchinson estimator,
  one-line modification to ISM. Mention as sidebar.
- **Live training in browser**: a "Train this score model" button
  that runs ~500 gradient steps on a 2D toy in real time. Heavy lift
  but immensely satisfying — closer to the VAE §8 stretch goal.
- **3D score field on a manifold**: a torus or sphere with a score
  field on its surface. Stretches the "low-dim manifold in high-dim
  space" intuition.
- **The reverse SDE perspective**: brief callout in §7 introducing
  the time-reversal formula for diffusion processes. Sets up
  score-based SDEs as a sidebar lesson.

---

## 10. Out of Scope (intentionally)

- **Score-based SDEs** (Song et al. 2021) — adjacent and worth a
  sidebar lesson, but the discrete-time view in §7 is sufficient
  setup for DDPM.
- **Flow matching** — adjacent generative-modeling paradigm; would
  warrant its own lesson after DDPM.
- **Energy-based models in full** — the EBM literature is large;
  mentioned in §2 as the origin of score matching but not developed.
- **Convergence theory for Langevin** — when does discretized
  Langevin actually converge? Mathematical machinery (log-Sobolev
  inequalities, Poincaré constants) is too deep for this lesson.
- **The connection between Fisher divergence and KL** (De Bruijn's
  identity) — mentioned in §3 as a footnote; not developed.

---

## 11. Training Notebook (offline pre-step)

The lesson ships pre-trained score-model weights for the §7
centerpiece. Training is done offline in Python. The notebook
(`docs/score-matching-training-notebook.ipynb`) should:

```python
# Pseudocode — agent expands and runs this
import numpy as np, torch, torch.nn as nn, json

# 1. Same 4-cluster 2D dataset as the VAE lesson (consistency)
np.random.seed(0)
centers = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2]])
N_per_cluster = 250
X = np.vstack([np.random.normal(c, 0.2, (N_per_cluster, 2)) for c in centers]).astype(np.float32)

# 2. Noise schedule (geometric)
sigma_min, sigma_max = 0.01, 2.0
L = 10
sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), L)).astype(np.float32)

# 3. Score network: takes (x, sigma) -> score in R^2
class ScoreNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        # Input: concat(x, log(sigma)) → 3 dims
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x, sigma):
        log_s = torch.log(sigma).unsqueeze(-1) if sigma.dim() == x.dim() - 1 else torch.log(sigma)
        return self.net(torch.cat([x, log_s], dim=-1))

# 4. DSM loss with sigma^2 weighting (so all noise levels have comparable scale)
def dsm_loss(model, x_clean, sigmas_batch):
    eps = torch.randn_like(x_clean)
    x_noisy = x_clean + sigmas_batch.unsqueeze(-1) * eps
    target = -eps / sigmas_batch.unsqueeze(-1)
    pred = model(x_noisy, sigmas_batch)
    # Weight by sigma^2 for balanced loss across noise levels
    return ((sigmas_batch.unsqueeze(-1) * (pred - target)) ** 2).sum(dim=-1).mean()

# 5. Train
model = ScoreNet(hidden=64)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xt = torch.tensor(X)
for epoch in range(10000):
    idx = np.random.choice(len(Xt), 128, replace=False)
    sigma_idx = np.random.choice(L, 128)
    sigmas_batch = torch.tensor(sigmas[sigma_idx])
    loss = dsm_loss(model, Xt[idx], sigmas_batch)
    opt.zero_grad(); loss.backward(); opt.step()

# 6. Save
weights = {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}
weights['_metadata'] = {
    'sigmas': sigmas.tolist(),
    'data_centers': centers.tolist(),
    'hidden_dim': 64,
}
with open('src/lessons/score-matching/assets/score-weights.json', 'w') as f:
    json.dump(weights, f)
```

The agent runs this notebook, inspects the resulting score field at
several noise levels (using the same `scoreSmoothedGMM` analytical
ground truth in §5's math module), and confirms qualitative match:
- At $\sigma = \sigma_L$ (smallest): learned field is sharp near the
  four cluster centers, weak elsewhere.
- At $\sigma = \sigma_1$ (largest): learned field is smooth, points
  globally toward the data centroid.
- Annealed Langevin starting from $\mathcal{N}(0, \sigma_1^2 I)$
  and running through the schedule produces ~100 samples that fall
  predominantly inside the four cluster envelopes.