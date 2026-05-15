# Denoising Diffusion Probabilistic Models (DDPM) — Interactive Lesson
## Build Specification & Content Plan

> **The destination lesson of the curriculum.** Every prior lesson —
> EM, KL & Jensen, ELBO/VI, the Gaussian Cookbook, VAE, Score
> Matching — converges here. By the end, the reader has read the Ho,
> Jain, Abbeel 2020 paper and understood every piece of it.
>
> DDPM is the largest, most visualization-rich lesson in StatViz. It
> is also the most ambitious, because it has to simultaneously:
>
> 1. Derive the variational bound in full.
> 2. Establish the genius ε-prediction parameterization.
> 3. Demonstrate the connection to score matching at multiple noise
>    levels — the paper's central conceptual contribution.
> 4. Make Algorithm 1 (training) and Algorithm 2 (sampling) feel
>    inevitable, not arbitrary.
> 5. Ship an interactive trained 2D DDPM that the reader can play
>    with for hours.
>
> The lesson sits at the **convergence node** of the roadmap. Two
> branches feed in:
>
> - **The variational branch**: KL → ELBO/VI → VAE. DDPM is a
>   hierarchical VAE with $T$ latent variables, a frozen encoder, and
>   a specific Gaussian forward structure.
> - **The score-matching branch**: KL → Gaussian Cookbook → Score
>   Matching. DDPM's simplified loss is denoising score matching at
>   $T$ noise levels; its sampler is annealed Langevin dynamics.
>
> The paper proves these are equivalent. The lesson makes the
> equivalence physical.
>
> Source: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion
> Probabilistic Models. NeurIPS 2020. (arXiv:2006.11239)

---

## 0. Pedagogical Philosophy

Same commitments as the rest of StatViz, dialed up:

1. **Two parallel framings, side by side.** Every section that
   introduces a quantity (the loss, the parameterization, the
   sampling step) gives both interpretations — variational and
   score-matching — and shows that they're the same object. This is
   what makes the paper hard to read in isolation, and what the
   lesson does best.
2. **No skipped algebra.** The derivations from the appendix of the
   paper (Equations 17–22 for the variance-reduced VLB; Equation 11
   for the ε-parameterization) appear in full. The paper itself
   leaves these as exercises; we don't.
3. **The trained 2D DDPM is the rhetorical climax.** Everything
   before §10 is buildup. §10 is the reader doing diffusion. Polish
   budget concentrates there.
4. **Every choice in the paper is justified or labeled as
   empirical.** When Ho et al. write "we found ε-prediction worked
   better," the lesson says so explicitly; it doesn't paper over
   choices that don't have first-principles answers.
5. **The connection back to every prior lesson is named.** EM's
   responsibilities show up in score visualization; VAE's
   reparameterization is the forward-process reparameterization;
   Score Matching's annealed Langevin is Algorithm 2 in disguise.
   Cross-link callouts saturate the lesson.

By the end, the reader can: (a) derive the variational bound for
DDPM from scratch, (b) state the closed-form $q(x_t \mid x_0)$
marginal and explain why it makes training tractable,
(c) derive the ε-prediction parameterization and explain why it
collapses the loss to simple MSE, (d) state both Algorithms 1 and 2
and trace each through a concrete example, (e) explain — in the
paper's framing — why DDPM is simultaneously variational inference
and denoising score matching, (f) predict qualitatively how changing
the noise schedule, the number of timesteps, or the $\Sigma_\theta$
choice affects sample quality, (g) read modern diffusion papers
(latent diffusion, classifier-free guidance, flow matching) using
DDPM as the shared baseline.

---

## 1. Tech Stack

Same Vite multi-page setup. Three considerations specific to this
lesson:

- **Browser-side DDPM inference.** The §10 trained model runs both
  the forward and reverse processes in the browser. Architecture is
  a small MLP with sinusoidal time embedding (same template as the
  Score Matching score network, with $t$ replacing $\sigma$). Forward
  passes are vanilla TS matrix multiplies via `ml-matrix`. No
  framework needed.
- **Animation budget.** Reverse sampling animates $T = 100$ steps
  for 100 particles per frame batch. That's $10{,}000$ forward
  passes per "Play" session, plus optional score-field overlay
  ($\sim 400$ more per frame refresh). All must run at $\geq 30$ fps.
  Architecture is sized accordingly — small enough that an unoptimized
  loop is fast enough.
- **Pre-trained weights as static asset.** Single trained model at
  $T = 100$, ships as ~50 KB JSON.

File layout:

```
src/lessons/ddpm/
├── main.ts
├── meta.ts
├── math/
│   ├── schedule.ts                    # betas, alphas, alpha_bars, tilde_betas
│   ├── forward-process.ts             # q(x_t | x_0), q(x_t | x_{t-1}), posterior
│   ├── reverse-process.ts             # mu_theta in eps-parameterization
│   ├── vlb.ts                         # variational bound decomposition
│   ├── score-conversion.ts            # eps_theta <-> s_theta
│   ├── eps-net.ts                     # the trained noise-prediction network
│   └── *.test.ts
├── sections/
│   ├── 01-hook.ts
│   ├── 02-forward-process.ts
│   ├── 03-reverse-process.ts
│   ├── 04-variational-bound.ts
│   ├── 05-forward-posterior.ts
│   ├── 06-parameterization.ts
│   ├── 07-score-matching-connection.ts
│   ├── 08-training-algorithm.ts
│   ├── 09-sampling-algorithm.ts
│   ├── 10-trained-ddpm-explorer.ts    # CENTERPIECE
│   ├── 11-practical-considerations.ts
│   └── 12-where-youll-see-this.ts
├── viz/
│   ├── hero-animation.ts              # §1 — opening visual
│   ├── forward-chain-viewer.ts        # §2 — forward diffusion animation
│   ├── closed-form-jump.ts            # §2 — one-shot vs iterative
│   ├── graphical-model.ts             # §3 — directed graph
│   ├── vlb-decomposition.ts           # §4 — three loss terms
│   ├── forward-posterior-explorer.ts  # §5 — q(x_{t-1} | x_t, x_0) interactive
│   ├── parameterization-comparison.ts # §6 — three parameterizations side-by-side
│   ├── score-equivalence.ts           # §7 — DDPM ↔ annealed Langevin
│   ├── training-trace.ts              # §8 — one training step animated
│   ├── sampling-chain.ts              # §9 — animated reverse process
│   ├── trained-ddpm-explorer.ts       # §10 — THE centerpiece
│   ├── coarse-to-fine-interp.ts       # §10 — Figure 9 recreation
│   ├── noise-schedule-explorer.ts     # §11 — schedule choice
│   └── final-roadmap.ts               # §12 — full curriculum graph
├── assets/
│   └── ddpm-weights.json              # pre-trained model
└── styles/
    └── overrides.css
```

---

## 2. Visual / Aesthetic Direction

Same paper-and-ink aesthetic, with the densest accent palette of any
lesson because DDPM has many distinct quantities:

```css
/* Data and noise */
--data-clean:     #b8651a;   /* x_0 — burnt sienna */
--data-noisy:     #c87f3b;   /* x_t at intermediate t — ochre */
--pure-noise:     #6b3a8c;   /* x_T — plum */

/* Processes */
--forward-arrow:  #5a8a6a;   /* forward process q — sage */
--reverse-arrow:  #2c5f8d;   /* reverse process p_theta — slate blue */
--posterior:      #d4a437;   /* the tractable posterior q(x_{t-1} | x_t, x_0) — amber */

/* Losses */
--loss-LT:        #999999;   /* L_T (small, gray) */
--loss-Lt:        #b54050;   /* L_{t-1} (main contributor, clay) */
--loss-L0:        #3a6b8c;   /* L_0 (reconstruction, blue) */

/* Score */
--score-vec:      #2c5f8d;   /* learned score arrows */
--score-target:   #b8651a;   /* DSM target (-eps/sigma) */
```

Visual conventions used throughout:
- The **forward chain** is drawn left-to-right with $x_0$ on the
  left, $x_T$ on the right. Forward arrows in `--forward-arrow`.
- The **reverse chain** is drawn right-to-left, with `--reverse-arrow`
  arrows. The two chains together produce the "diamond" graph from
  Figure 2 of the paper.
- $\hat x_0$ **estimates** (the model's running belief about clean
  data at any timestep) are drawn in `--data-clean` outline with a
  fade indicating confidence.
- **Score-field overlays** (when toggled) use the same arrow
  convention as the Score Matching lesson — `--score-vec`, scaled to
  ~5% canvas width for the longest vector.

---

## 3. Lesson Metadata (`src/lessons/ddpm/meta.ts`)

```ts
import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'ddpm',
  title: 'Denoising Diffusion Probabilistic Models',
  subtitle: 'A hierarchical VAE, a score-matching model, and an annealed Langevin sampler — all the same thing.',
  tier: 4,
  difficulty: 4,
  estimatedHours: 6,
  status: 'planned',
  prerequisites: [
    { id: 'vae',              strength: 'required',    anchor: 'vae-objective' },
    { id: 'score-matching',   strength: 'required',    anchor: 'dsm' },
    { id: 'gaussian-cookbook',strength: 'required',    anchor: 'conditioning' },
    { id: 'elbo-vi',          strength: 'required',    anchor: 'elbo-two-forms' },
    { id: 'kl-jensen',        strength: 'required',    anchor: 'kl-gaussians' },
    { id: 'em',               strength: 'recommended', anchor: 'q-function' },
  ],
  recommendedNext: [],
  alsoUsedBy: [],
  description:
    'Denoising diffusion probabilistic models: a hierarchical latent ' +
    'variable model with fixed Gaussian forward process and learned ' +
    'reverse Markov chain. Trained as variational inference, with the ' +
    'epsilon-prediction parameterization revealing equivalence to ' +
    'denoising score matching at T noise levels and annealed Langevin ' +
    'sampling. The destination of the StatViz curriculum.',
  exportedAnchors: {
    'forward-process':       'The forward diffusion process (q)',
    'closed-form-marginal':  'The closed-form marginal q(x_t | x_0)',
    'reverse-process':       'The reverse process (p_theta) as a hierarchical VAE',
    'vlb':                   'The variational lower bound and its decomposition',
    'forward-posterior':     'The tractable forward posterior q(x_{t-1} | x_t, x_0)',
    'eps-parameterization':  'The epsilon-prediction parameterization',
    'sm-equivalence':        'The denoising score matching equivalence',
    'training-algorithm':    'Algorithm 1: training',
    'sampling-algorithm':    'Algorithm 2: sampling',
    'L-simple':              'The simplified training objective',
  },
  path: '/lessons/ddpm',
};
```

---

## 4. Section-by-Section Plan

Twelve sections. Plan for ~3–4 hours of focused reading + interaction.
This is the deepest lesson in the curriculum.

---

### Section 1 — Hook

**Length**: ~250 words.

**Prose** (verbatim):

> Diffusion models do something that, the first time you see it,
> looks like a magic trick. Start with pure Gaussian noise — a
> canvas of static. Apply a neural network 1000 times in sequence,
> each time nudging the image slightly. At the end, the static has
> become a photorealistic human face that has never existed.
>
> The training story is just as strange. The model is never shown a
> single complete image alongside a goal. It only ever sees a real
> image with some Gaussian noise added, and it's asked: *what noise
> was added?* That's the entire training signal.
>
> How does this work? **Two answers, both true.**
>
> First answer (variational): A diffusion model is a hierarchical
> VAE. It has $T = 1000$ latent variables $x_1, \ldots, x_T$,
> arranged in a Markov chain. The "encoder" is frozen: it just adds
> Gaussian noise at each step until pure noise remains. The "decoder"
> — the reverse Markov chain $p_\theta(x_{t-1} \mid x_t)$ — is what
> gets learned, by gradient descent on the ELBO.
>
> Second answer (score matching): A diffusion model learns
> $\nabla \log p(x_t)$ at every noise level $t$, then samples by
> walking down the noise schedule with Langevin dynamics. The
> training loss is denoising score matching. The sampling procedure
> is annealed Langevin.
>
> **Both answers describe the same object.** The paper's central
> contribution (Ho et al. 2020) is showing that the right
> parameterization of the variational model makes it literally
> equivalent to the score model. This lesson develops both views in
> parallel and lands on the equivalence as a punchline.

CTA button: "Watch the forward process →"

**Visualization 1 — `<HeroAnimation>`** (full width, hero size):

A 2D synthetic-data animation. Two stripes:
- **Top stripe**: data clusters at $t = 0$, gradually diffusing into
  pure Gaussian noise as $t$ increases left to right. Eight snapshot
  panels.
- **Bottom stripe**: the reverse — pure noise at the left, gradually
  un-diffusing into data clusters as $t$ decreases (or as one walks
  right). Eight snapshot panels.

The eight panels in each stripe correspond to $t = 0, 14, 29, 43, 57,
71, 86, 100$ (for the trained model at $T = 100$). Bottom panels
show **what the trained DDPM produces** at each step — actual model
output, not simulated.

This is the lesson in one image. Auto-loops every 8 seconds.

---

### Section 2 — The Forward Process: Destroying Signal

**Length**: ~900 words. **Anchors: `forward-process`, `closed-form-marginal`**.

**Prose**:

> The **forward process** (also called the **diffusion process**) is
> a fixed Markov chain that gradually adds Gaussian noise to data
> until nothing recognizable remains. Define it by:
>
> $$\boxed{\;\; q(x_t \mid x_{t-1}) \;=\; \mathcal{N}\!\big(x_t; \;\sqrt{1 - \beta_t} \, x_{t-1}, \; \beta_t I\big) \;\;}$$
>
> where $\beta_1, \beta_2, \ldots, \beta_T \in (0, 1)$ is a small
> sequence of variances (the **noise schedule**). At each step:
> shrink the previous sample by $\sqrt{1 - \beta_t}$ and add Gaussian
> noise with variance $\beta_t$. The full forward distribution is:
>
> $$q(x_{1:T} \mid x_0) \;=\; \prod_{t=1}^{T} q(x_t \mid x_{t-1})$$
>
> Two design choices worth dissecting.
>
> #### Why the $\sqrt{1 - \beta_t}$ scaling
>
> Without the scaling factor, variance would *accumulate*: each step
> adds $\beta_t$ of variance on top of whatever variance the
> previous $x_{t-1}$ had. After many steps, $\mathrm{Var}(x_t)$ would
> grow without bound.
>
> With the scaling: if $\mathrm{Var}(x_{t-1}) = 1$, then
> $\mathrm{Var}(x_t) = (1 - \beta_t) \cdot 1 + \beta_t = 1$. **Variance
> is preserved.** A unit-variance signal $x_0$ stays approximately
> unit-variance throughout the chain. This is essential because the
> reverse process eventually has to start from $\mathcal{N}(0, I)$
> — a unit-variance distribution — so the chain endpoints have to
> match.
>
> The paper's exact schedule: $T = 1000$, $\beta_t$ linear from
> $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. Small at the start (so
> early steps preserve fine detail) and slightly larger at the end
> (so the chain reaches near-Gaussian noise by $T$).

> #### The closed-form jump
>
> The remarkable property of the forward process is that **we can
> sample $x_t$ from $x_0$ directly, without iterating through the
> intermediate steps.** Define:
>
> $$\alpha_t \;:=\; 1 - \beta_t, \qquad \bar\alpha_t \;:=\; \prod_{s=1}^{t} \alpha_s$$
>
> Then:
>
> $$\boxed{\;\; q(x_t \mid x_0) \;=\; \mathcal{N}\!\big(x_t; \; \sqrt{\bar\alpha_t} \, x_0, \; (1 - \bar\alpha_t) I\big) \;\;}$$
>
> One Gaussian, available in closed form. This is the single most
> important property of the forward process — without it, training
> would require an $O(T)$ inner loop per data point.
>
> **Derivation**: by induction. The base case $t = 1$ is the
> definition. For the inductive step, assume
> $x_{t-1} = \sqrt{\bar\alpha_{t-1}} \, x_0 + \sqrt{1 - \bar\alpha_{t-1}} \, \tilde\epsilon$
> for $\tilde\epsilon \sim \mathcal{N}(0, I)$. Then
>
> $$x_t \;=\; \sqrt{1 - \beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon \;=\; \sqrt{\alpha_t \bar\alpha_{t-1}} \, x_0 + \sqrt{\alpha_t (1 - \bar\alpha_{t-1})} \, \tilde\epsilon + \sqrt{\beta_t} \, \epsilon$$
>
> The two independent Gaussian noise contributions combine into a
> single Gaussian of variance $\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t = \alpha_t - \alpha_t \bar\alpha_{t-1} + 1 - \alpha_t = 1 - \bar\alpha_t$.
> So $x_t = \sqrt{\bar\alpha_t} \, x_0 + \sqrt{1 - \bar\alpha_t} \, \epsilon'$
> for some standard Gaussian $\epsilon'$. $\blacksquare$

> #### Reading the marginal
>
> $$x_t \;=\; \sqrt{\bar\alpha_t} \, x_0 \;+\; \sqrt{1 - \bar\alpha_t} \, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$
>
> Two terms:
> - $\sqrt{\bar\alpha_t} \, x_0$: the **signal** — a shrinking copy
>   of the original data.
> - $\sqrt{1 - \bar\alpha_t} \, \epsilon$: the **noise** — a growing
>   standard Gaussian contribution.
>
> The signal-to-noise ratio is $\bar\alpha_t / (1 - \bar\alpha_t)$,
> which **decreases monotonically with $t$**. At $t = 0$: pure signal.
> At $t = T$: pure noise.

> #### Worked numerical values
>
> For the paper's schedule ($T = 1000$, $\beta_t$ linear from $10^{-4}$
> to $0.02$):
>
> | $t$ | $\beta_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ | $\sqrt{1 - \bar\alpha_t}$ |
> |:---:|:---------:|:--------------:|:---------------------:|:-------------------------:|
> | 1   | $10^{-4}$ | $0.9998$       | $0.9999$              | $0.0148$ |
> | 100 | $0.0021$  | $0.895$        | $0.946$               | $0.324$ |
> | 250 | $0.0051$  | $0.521$        | $0.722$               | $0.692$ |
> | 500 | $0.0101$  | $0.0778$       | $0.279$               | $0.960$ |
> | 750 | $0.0150$  | $0.0033$       | $0.057$               | $0.998$ |
> | 999 | $0.0200$  | $4 \cdot 10^{-5}$ | $0.006$            | $1.000$ |
>
> Note: by $t = 500$, **less than 8% of the original signal
> variance survives.** By $t = 1000$, less than $4 \cdot 10^{-3}$ %
> — the data is essentially pure standard normal noise.
>
> (Pre-verified; test target in `schedule.test.ts`.)

**Cross-link callout — back to Gaussian Cookbook** (`type=back`):

> **Uses: [Gaussian Cookbook §4 — reparameterization](../gaussian-cookbook/#reparam-matrix)**.
> The closed-form $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$
> is the reparameterization trick: $\sqrt{\bar\alpha_t} x_0$ acts as
> the mean, $\sqrt{1 - \bar\alpha_t}$ as the scale, $\epsilon$ as the
> noise. Two affine-transformed Gaussians compose into one Gaussian
> with the matching mean and variance.

**Visualization 2 — `<ForwardChainViewer>`** (full width):

A 2D canvas showing the 4-cluster synthetic dataset at $t = 0$.
A slider for $t \in [0, T]$ controls the timestep. As the user
drags:
- Background: 1000 training data points morph from their clean
  positions to a near-isotropic Gaussian distribution.
- Overlay: a single highlighted data point ($x_0$) with its current
  noisy version $x_t$ tracked through the chain.
- Annotation: the current $\sqrt{\bar\alpha_t}$ (signal strength)
  and $\sqrt{1 - \bar\alpha_t}$ (noise strength) as numbers.
- The marginal $q(x_t \mid x_0)$ drawn as an ellipse around $x_t$
  showing its variance.

Buttons:
- "Animate forward" — slides $t$ from 0 to $T$ over 10 seconds.
- "Animate back" — slides from $T$ to 0 (note: this is using $q$,
  not the reverse process — it's just visualizing the slider going
  backward, useful for inspection).
- "Re-roll noise" — generates fresh $\epsilon$ for the highlighted
  point.

The visualization makes the **destruction of signal** physical. By
$t = 500$, clusters have visually merged. By $t = 1000$, they're
indistinguishable from $\mathcal{N}(0, I)$.

**Visualization 3 — `<ClosedFormJump>`** (medium width):

A focused demonstration. Two side-by-side animations:
- Left: iterative sampling — start at $x_0$, apply $q(x_t \mid x_{t-1})$
  $t$ times in sequence. The animation visibly takes $t$ time steps.
- Right: closed-form jump — apply $q(x_t \mid x_0)$ directly. One
  instant.

Both produce samples from the same distribution. The user picks
$t \in \{50, 250, 500, 999\}$; both panels jump to that $t$. The
right panel finishes immediately; the left takes (artificially) one
second per step.

Annotation: "Training requires sampling $x_t$ at random timesteps
for millions of data points. Without the closed-form jump, this
would be infeasible."

---

### Section 3 — The Reverse Process: A Hierarchical VAE

**Length**: ~700 words. **Anchor: `reverse-process`**.

**Prose**:

> The **reverse process** is what we learn. Define it as a Markov
> chain starting at the standard Gaussian and working backward:
>
> $$\boxed{\;\; p_\theta(x_{0:T}) \;=\; p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t), \qquad p(x_T) \;=\; \mathcal{N}(x_T; 0, I) \;\;}$$
>
> Each reverse transition is parameterized as a Gaussian:
>
> $$p_\theta(x_{t-1} \mid x_t) \;=\; \mathcal{N}\!\big(x_{t-1}; \; \mu_\theta(x_t, t), \; \Sigma_\theta(x_t, t)\big)$$
>
> The neural network takes the current noisy sample $x_t$ and the
> timestep $t$, and outputs the mean (and possibly variance) of the
> Gaussian distribution for the previous step. To sample data, start
> at $x_T \sim \mathcal{N}(0, I)$ and walk backward through the chain.

> #### Why Gaussian reverse transitions work
>
> A subtle and important fact: when the forward step variance $\beta_t$
> is **small**, the reverse transition $q(x_{t-1} \mid x_t)$ is
> approximately Gaussian — even though it's not Gaussian in general.
> This is a classical result from Feller (1949), referenced in
> Sohl-Dickstein et al. (2015). The paper exploits this: small
> $\beta_t$ means the forward and reverse processes have **the same
> functional form** (Gaussian), justifying the parameterization
> choice.
>
> If $\beta_t$ were large, the reverse step would be highly
> multimodal (which mode of the data did $x_t$ come from?) and a
> Gaussian wouldn't fit. With $\beta_t \leq 0.02$, the local
> Gaussian approximation is tight enough to drive a working sampler.

> #### Reading DDPM as a VAE
>
> Compare to [VAE](../vae/) where we had:
>
> - Encoder $q_\phi(z \mid x)$ — learned, neural network.
> - Single latent $z$.
> - Decoder $p_\theta(x \mid z)$ — learned.
>
> DDPM:
>
> - Encoder $q(x_{1:T} \mid x_0)$ — **fixed**, no learnable
>   parameters (the noise schedule is the only "structure").
> - **$T$ latents** $x_1, x_2, \ldots, x_T$, of the **same
>   dimensionality** as the data (this is crucial — unlike VAE where
>   $z$ is low-dimensional).
> - Decoder $p_\theta(x_0 \mid x_T) = \int p(x_T) \prod p_\theta(x_{t-1} \mid x_t) dx_{1:T-1}$
>   — learned, parameterized as a chain.
>
> **DDPM is a hierarchical VAE with a frozen encoder and $T$ same-dim
> latents arranged in a Markov chain.** Everything you know about
> VAE training — the ELBO, the reparameterization trick, the
> variational objective — generalizes directly.
>
> The frozen encoder is what makes DDPM both more constrained and
> more powerful than a standard VAE. More constrained: we can't
> learn what kind of latent code is most useful — it's fixed to be
> "noisy versions of the data." More powerful: the constraint is
> the *right* constraint for image data, where additive Gaussian
> noise is geometrically natural.

**Cross-link callout — back to VAE** (`type=back`):

> **Uses: [VAE](../vae/) framework**. Everything below — the ELBO
> derivation, the reparameterization trick, the gradient-of-log-prob
> objective — extends VAE to a $T$-step hierarchical setting with a
> frozen encoder. If §4–§6 feel like VAE déjà vu, that's because
> they are.

**Visualization 4 — `<GraphicalModel>`** (medium width):

A static-but-interactive recreation of Figure 2 from the paper. The
directed graphical model $x_T \to x_{T-1} \to \cdots \to x_0$
with reverse-process arrows in `--reverse-arrow`, plus the
**dashed** $q(x_t \mid x_{t-1})$ arrows going the opposite direction
in `--forward-arrow`. Each node shows a small thumbnail of an
image (or a 2D point) at that noise level.

Hover any node: see the marginal $q(x_t \mid x_0)$ described in
plain text ("at $t = 500$, the signal is at 28% of original strength
and the noise has 96% of total variance").

Hover any arrow: see the formula and direction in plain text.

The graphical model is the most-deep-link-target in the lesson:
modern diffusion papers (and this very lesson's own §11) will refer
to "the $x_t \to x_{t-1}$ transition" without re-explaining. This
viz is the persistent reference.

---

### Section 4 — The Variational Bound

**Length**: ~900 words. **Anchor: `vlb`**.

**Prose**:

> Now derive the loss. We want to maximize $\log p_\theta(x_0)$ for
> data $x_0 \sim q(x_0)$ (where $q(x_0) = p_{\text{data}}$ is the
> data distribution). Because $p_\theta(x_0) = \int p_\theta(x_{0:T}) dx_{1:T}$
> is intractable, we use the variational lower bound — the standard
> VAE trick, scaled to $T$ latents.

> #### The standard ELBO
>
> Using $q(x_{1:T} \mid x_0)$ as the variational distribution (it's
> not actually variational here since it's fixed, but it plays the
> same role):
>
> $$-\log p_\theta(x_0) \;\leq\; \mathbb{E}_q\!\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] \;=:\; L$$
>
> This is the negative-ELBO form of Form 1 from
> [ELBO/VI §4](../elbo-vi/#elbo-two-forms). Expand by definition of
> the products:
>
> $$L \;=\; \mathbb{E}_q\!\left[-\log p(x_T) \;-\; \sum_{t \geq 1} \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}\right]$$
>
> A sum of $T + 1$ terms. We could train on this directly with
> Monte Carlo, but each term has high variance because $q(x_t \mid x_{t-1})$
> is sharply peaked. **The paper's first algebraic trick is to
> rewrite this in a form where every term is a KL between two
> Gaussians** — which can be computed in closed form (using the
> Cookbook), eliminating the Monte Carlo variance from those terms.

> #### The variance-reduced form
>
> $$\boxed{\;\; L \;=\; \mathbb{E}_q\!\left[\underbrace{D_{\mathrm{KL}}\!\big(q(x_T \mid x_0) \,\|\, p(x_T)\big)}_{L_T} \;+\; \sum_{t > 1} \underbrace{D_{\mathrm{KL}}\!\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big)}_{L_{t-1}} \;\underbrace{-\log p_\theta(x_0 \mid x_1)}_{L_0}\right] \;\;}$$
>
> Three families of terms:
>
> - $L_T$: KL between the end of the forward chain and the prior.
>   With $\beta_t$ chosen so the forward chain reaches near-Gaussian
>   noise, this is **tiny** ($\approx 2.9 \times 10^{-5}$ bits/dim
>   for the paper's setup — pre-verified) and **constant in $\theta$**
>   (the forward process has no parameters). Drop it.
> - $L_{t-1}$ for $t = 2, \ldots, T$: KL between the **tractable
>   forward posterior** $q(x_{t-1} \mid x_t, x_0)$ (which we'll
>   compute in §5) and the **learned reverse transition**
>   $p_\theta(x_{t-1} \mid x_t)$. Both are Gaussian → closed-form KL
>   via the [Gaussian Cookbook](../gaussian-cookbook/#kl-mvn).
> - $L_0$: the final reconstruction term. Different from the others
>   because the data is discrete (image pixels in $\{0, \ldots, 255\}$);
>   the paper handles this with a discretized Gaussian decoder
>   (Equation 13 of the paper).

> #### Why this is variance-reduced
>
> In the original sum, each term involved $\log p_\theta(x_{t-1} \mid x_t) - \log q(x_t \mid x_{t-1})$,
> which is the *ratio* of two Gaussians evaluated at samples drawn
> from a third. Monte Carlo estimates of such ratios have variance
> that grows with how different the distributions are.
>
> The rewritten form replaces these high-variance log-ratios with
> **closed-form KL divergences between Gaussians**. The expected
> values are computed analytically; the Monte Carlo only needs to
> sample $x_0$ and the noise (one sample per gradient step gives
> good signal). This is what makes training viable.

> #### The full derivation (in `<ProofToggle>`, expanded by default)
>
> Start from $L = \mathbb{E}_q[-\log p_\theta(x_{0:T}) / q(x_{1:T} \mid x_0)]$.
> Expand the logs and rearrange:
>
> $$L \;=\; \mathbb{E}_q\!\left[-\log p(x_T) \;-\; \sum_{t > 1}\log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})} \;-\; \log \frac{p_\theta(x_0 \mid x_1)}{q(x_1 \mid x_0)}\right]$$
>
> The key move: by Bayes' rule with $x_0$ included,
>
> $$q(x_t \mid x_{t-1}) \;=\; q(x_t \mid x_{t-1}, x_0) \;=\; \frac{q(x_{t-1} \mid x_t, x_0) \, q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$$
>
> (The first equality is the Markov property — $x_t$ given $x_{t-1}$
> doesn't depend on $x_0$.) Substituting this into the sum:
>
> $$\sum_{t > 1} \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})} \;=\; \sum_{t > 1} \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_{t-1} \mid x_t, x_0)} \;+\; \sum_{t > 1} \log \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$
>
> The second sum **telescopes**:
>
> $$\sum_{t > 1} \log \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} \;=\; \log \frac{q(x_1 \mid x_0)}{q(x_T \mid x_0)}$$
>
> Plug everything back and the $\log q(x_1 \mid x_0)$ cancels with
> the trailing term. After grouping:
>
> $$L \;=\; \mathbb{E}_q\!\left[\log \frac{q(x_T \mid x_0)}{p(x_T)} \;+\; \sum_{t > 1} \log \frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)} \;-\; \log p_\theta(x_0 \mid x_1)\right]$$
>
> The first two pieces are KL divergences (under expectation over $q$,
> $\mathbb{E}_q[\log q/p] = D_{\mathrm{KL}}(q \| p)$). The third is
> the reconstruction term. We arrive at the boxed decomposition. $\blacksquare$
>
> (See Appendix A of the paper for the same derivation.)

**Visualization 5 — `<VLBDecomposition>`** (medium width):

A stacked horizontal bar visualizing the three components of $L$ for
the paper's setup at $T = 1000$:
- $L_T$ rendered as a thin sliver in `--loss-LT` (essentially zero).
- $L_0$ as a small block in `--loss-L0`.
- The sum $\sum L_{t-1}$ as the dominant block in `--loss-Lt`.

Below the main bar: a per-timestep breakdown of $L_{t-1}$ as a line
chart showing the weighting (Eq 12 of the paper) — which we'll
discuss in §6. The user can already see the qualitative shape
(weight peaks at small $t$).

Annotations point at:
- The dominant contribution of $L_{t-1}$: "this is the loss we'll
  spend most of §5–§8 understanding."
- The dropping of $L_T$: "constant in $\theta$, contributes nothing
  to gradients."
- The special role of $L_0$: "discretized Gaussian decoder — needed
  for proper log-likelihood, simplified away in $L_{\text{simple}}$."

---

### Section 5 — The Tractable Forward Posterior

**Length**: ~700 words. **Anchor: `forward-posterior`**.

**Prose**:

> The key new object is $q(x_{t-1} \mid x_t, x_0)$ — the
> distribution of $x_{t-1}$ conditional on both $x_t$ (where we are
> now in the chain) and $x_0$ (the original clean data).
>
> Why include $x_0$? Without it, $q(x_{t-1} \mid x_t)$ is the
> distribution we'd love to compute — the "true reverse step." But
> it's intractable: it's a function of the entire data distribution.
>
> **With $x_0$ included, the posterior is Gaussian and closed-form.**

> #### The closed-form posterior
>
> Conditional on $x_0$ and $x_t$, the joint $(x_{t-1}, x_t \mid x_0)$
> is jointly Gaussian (because both marginals and the conditional are
> Gaussian — chained Gaussians). By the
> [conditioning identity](../gaussian-cookbook/#conditioning):
>
> $$\boxed{\;\; q(x_{t-1} \mid x_t, x_0) \;=\; \mathcal{N}\!\big(x_{t-1}; \; \tilde\mu_t(x_t, x_0), \; \tilde\beta_t I\big) \;\;}$$
>
> where
>
> $$\tilde\mu_t(x_t, x_0) \;=\; \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 \;+\; \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t$$
>
> $$\tilde\beta_t \;=\; \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$$

> #### Reading the formulas
>
> The posterior mean $\tilde\mu_t$ is a **convex combination** of
> $x_0$ and $x_t$:
>
> $$\tilde\mu_t \;=\; w_t^{(0)} x_0 + w_t^{(t)} x_t, \qquad w_t^{(0)} + w_t^{(t)} \cdot \frac{1}{\sqrt{\alpha_t}} \;\approx\; 1$$
>
> (Not exactly convex — there's a $\sqrt{\alpha_t}$ factor — but
> close.) The weights interpolate:
> - At **small $t$**: $w_t^{(t)}$ dominates. The posterior mean is
>   essentially $x_t / \sqrt{\alpha_t}$. We trust the current noisy
>   sample, since it has very little noise.
> - At **large $t$**: $w_t^{(0)}$ dominates. The posterior mean is
>   pulled toward $x_0$. The current noisy sample is essentially
>   pure noise; we have to lean on the clean data.
> - At **intermediate $t$**: a graded mixture.
>
> The posterior variance $\tilde\beta_t$ is a **rescaled version** of
> the forward step variance $\beta_t$:
>
> $$\tilde\beta_t \;\leq\; \beta_t$$
>
> Why ≤? Because conditioning on $x_0$ resolves uncertainty about
> $x_{t-1}$ — the conditional variance is smaller than the marginal
> step variance. At $t = 1$ (lots of resolution from $x_0$),
> $\tilde\beta_1 / \beta_1 \approx 0.45$. At large $t$ (where $x_0$
> tells us little), $\tilde\beta_t / \beta_t \to 1$.

> #### What this gives us
>
> The forward posterior is the **target distribution** that the
> reverse process $p_\theta(x_{t-1} \mid x_t)$ tries to match. The
> $L_{t-1}$ term in the loss is:
>
> $$L_{t-1} \;=\; D_{\mathrm{KL}}\!\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big)$$
>
> Both distributions are Gaussian. If we parameterize $p_\theta$
> with the same variance as $q$ (i.e., $\Sigma_\theta = \tilde\beta_t I$
> or $\beta_t I$), then by [the shared-covariance special case of
> Gaussian KL](../gaussian-cookbook/#kl-mvn):
>
> $$L_{t-1} \;=\; \frac{1}{2 \sigma_t^2} \|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2 \;+\; \text{const}$$
>
> The loss reduces to mean-squared error between the model's
> predicted mean and the analytical posterior mean. **This is the
> setup for §6's parameterization choice.**

**Cross-link callout — back to Gaussian Cookbook** (`type=back`):

> **Uses: [Gaussian Cookbook §5 — conditioning](../gaussian-cookbook/#conditioning)**
> and [§3 — shared-covariance KL](../gaussian-cookbook/#kl-mvn). The
> jointly-Gaussian conditioning identity gives the closed-form
> posterior. The shared-covariance KL reduces $L_{t-1}$ to squared
> Mahalanobis distance between means.

**Visualization 6 — `<ForwardPosteriorExplorer>`** (full width):

A 2D canvas with three plotted points:
- $x_0$ (in `--data-clean`, draggable).
- $x_t$ (in `--data-noisy`, draggable).
- $\tilde\mu_t$ (in `--posterior`, computed and rendered).

Plus an ellipse around $\tilde\mu_t$ showing $\tilde\beta_t I$.

Slider: $t \in [1, T]$. As the user moves $t$:
- $\tilde\mu_t$ moves along the line from $x_t$ (at small $t$)
  toward $x_0$ (at large $t$).
- The ellipse shrinks at small $t$, grows at large $t$.

Three modes:
- "Drag $x_t$" — fix $x_0$ at a data cluster, drag $x_t$ around the
  canvas, watch the posterior follow.
- "Fix $x_t$" — fix $x_t$, move $x_0$ to see which $x_0$ "explains"
  this $x_t$.
- "Trajectory" — animate $x_t$ along a typical forward-process
  trajectory from $x_0$, with the posterior tracking back toward $x_0$
  as $t$ decreases. **This is the visualization that makes the
  posterior feel natural.**

---

### Section 6 — Parameterizing the Reverse Process

**Length**: ~1100 words. **Anchor: `eps-parameterization`**. The key
algebraic trick of the paper.

**Prose**:

> The variational bound, post-decomposition, says: train
> $\mu_\theta(x_t, t)$ to match $\tilde\mu_t(x_t, x_0)$ (with $x_0$
> in the expectation). How should we parameterize $\mu_\theta$?
>
> Three natural choices, **mathematically equivalent**:
>
> 1. Predict the posterior mean directly: $\mu_\theta(x_t, t) \approx \tilde\mu_t$.
> 2. Predict $x_0$ from $x_t$, then plug into $\tilde\mu_t$:
>    $\mu_\theta(x_t, t) = \tilde\mu_t(x_t, x_0^{\theta}(x_t))$.
> 3. Predict the noise $\epsilon$ from $x_t$, then back out
>    $\mu_\theta$.
>
> All three lead to the same optimal $\mu_\theta^*$. They differ in
> **what the network is asked to output** — and this affects
> optimization dynamics significantly. The paper finds (and shows
> empirically in Table 2) that **option 3 — predicting $\epsilon$ —
> works best.**

> #### Deriving the $\epsilon$-prediction parameterization
>
> Start from the marginal $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$.
> Solve for $x_0$:
>
> $$x_0 \;=\; \frac{1}{\sqrt{\bar\alpha_t}}\!\left(x_t - \sqrt{1 - \bar\alpha_t} \, \epsilon\right)$$
>
> Substitute into the posterior mean $\tilde\mu_t(x_t, x_0)$ from §5
> and simplify. After algebra (the coefficient of $x_t$ collapses to
> $1/\sqrt{\alpha_t}$ via $\alpha_t + \beta_t = 1$ and
> $\bar\alpha_t = \alpha_t \bar\alpha_{t-1}$):
>
> $$\boxed{\;\; \tilde\mu_t(x_t, x_0(\epsilon, x_t)) \;=\; \frac{1}{\sqrt{\alpha_t}}\!\left(x_t \;-\; \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon\right) \;\;}$$
>
> **The posterior mean, in terms of $x_t$ and the noise $\epsilon$
> that produced it, has a remarkably clean form.** Two terms:
> - $x_t / \sqrt{\alpha_t}$: the current sample, slightly amplified.
> - $-\beta_t \, \epsilon / (\sqrt{\alpha_t} \sqrt{1 - \bar\alpha_t})$:
>   a correction proportional to the noise.
>
> The reverse-process mean parameterization is then:
>
> $$\boxed{\;\; \mu_\theta(x_t, t) \;=\; \frac{1}{\sqrt{\alpha_t}}\!\left(x_t \;-\; \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(x_t, t)\right) \;\;}$$
>
> The network $\epsilon_\theta(x_t, t)$ outputs a vector of the same
> shape as $x_t$, interpreted as the model's estimate of "what noise
> was added to produce $x_t$."

> #### What the loss becomes
>
> Substitute the $\epsilon$-parameterized $\mu_\theta$ into
> $L_{t-1} = \frac{1}{2\sigma_t^2} \|\tilde\mu_t - \mu_\theta\|^2$.
> The $1/\sqrt{\alpha_t}$ and $\beta_t/\sqrt{1 - \bar\alpha_t}$
> factors come out as a per-$t$ constant:
>
> $$L_{t-1} \;=\; \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar\alpha_t)} \, \mathbb{E}_{x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta\!\big(\sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \;t\big)\big\|^2\right]$$
>
> (Pre-verified to match Equation 12 of the paper.)
>
> **The model is asked to predict $\epsilon$, the noise that was
> added.** The loss is MSE on noise prediction, weighted by a
> per-timestep factor.
>
> If we set $\sigma_t^2 = \beta_t$ (one of the two paper choices) and
> drop all per-$t$ weights:
>
> $$L_{\text{simple}}(\theta) \;=\; \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta\!\big(\sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \;t\big)\big\|^2\right]$$
>
> (Equation 14 of the paper.) The whole loss collapses to: **predict
> the noise from the noisy sample.** Eight lines of code.

> #### Why $\epsilon$-prediction works better
>
> The paper's empirical result (Table 2): predicting $\epsilon$ with
> $L_{\text{simple}}$ gives FID 3.17 on CIFAR10; predicting $\tilde\mu$
> with $L_{\text{simple}}$ doesn't converge.
>
> Three plausible reasons:
> 1. **Scale stability**. $\epsilon$ has unit variance for all $t$
>    (it's standard Gaussian by construction). $\tilde\mu_t$ has
>    variance that scales with $\bar\alpha_t$ — small at large $t$.
>    A unit-variance target lets the same network width and learning
>    rate work across all $t$.
> 2. **The down-weighting of small-$t$ terms by dropping the
>    weight**. The paper-derived weight $\beta_t^2 / (2 \sigma_t^2 \alpha_t (1 - \bar\alpha_t))$
>    is **larger at small $t$** (where the loss is "easier" — almost-clean
>    data). $L_{\text{simple}}$ flattens this. The intuition: focus
>    capacity on harder denoising at large $t$, where the model
>    actually has to *generate* structure.
> 3. **Implicit connection to score matching**. With $\epsilon$-prediction,
>    the loss is literally denoising score matching (next section).
>    The optimization is well-conditioned because the target
>    geometry is the score field.

> #### Numerical example
>
> At $t = 500$ in the paper's schedule: $\beta_t = 0.0101$,
> $\alpha_t = 0.9899$, $\bar\alpha_t = 0.078$. With $\sigma_t^2 = \beta_t$:
>
> $$\text{weight}_{500} \;=\; \frac{(0.0101)^2}{2 \cdot 0.0101 \cdot 0.9899 \cdot 0.922} \;\approx\; 0.0055$$
>
> Compare to $t = 1$:
>
> $$\text{weight}_{1} \;=\; \frac{(10^{-4})^2}{2 \cdot 10^{-4} \cdot 0.9999 \cdot 2 \cdot 10^{-4}} \;\approx\; 0.25$$
>
> The weight at $t = 1$ is **~45× the weight at $t = 500$**. Under
> the full bound, the model is heavily pushed to be accurate on
> easy-denoising tasks. $L_{\text{simple}}$ undoes this. Verified;
> test target in `vlb.test.ts`.

**Visualization 7 — `<ParameterizationComparison>`** (full width):

Three small panels in a row showing the same trained-DDPM
intermediate state ($t = 500$, $x_t$ fixed) under three different
output interpretations:
- **Panel A**: $\mu_\theta(x_t, t)$ — direct prediction of posterior
  mean. The output is in data space.
- **Panel B**: $x_0^\theta(x_t, t)$ — prediction of clean data. The
  output is also in data space but mapped backward.
- **Panel C**: $\epsilon_\theta(x_t, t)$ — prediction of noise. The
  output is in noise space, ~unit-variance.

The user sees: all three panels' outputs are **mathematically
equivalent** (each is derivable from the others via the formulas in
this section). But the **scales** are different — A and B have
output magnitudes that scale with $\bar\alpha_t$, while C has roughly
unit magnitude. A small histogram below each panel makes this
visible.

Annotation: "Network outputs are mathematically equivalent. But the
loss landscape under MSE differs — the C parameterization has the
most stable scale across $t$."

---

### Section 7 — The Score Matching Connection

**Length**: ~900 words. **Anchor: `sm-equivalence`**. **The paper's
central conceptual contribution.**

**Prose**:

> We've derived $L_{\text{simple}}$ from variational principles.
> Now we'll show — exactly as Ho et al. show in §3.2 of the paper —
> that the same loss arises from **denoising score matching** at $T$
> different noise levels.
>
> Read the simplified loss again:
>
> $$L_{\text{simple}}(\theta) \;=\; \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2\right], \quad x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$$
>
> Compare to denoising score matching from
> [Score Matching §5](../score-matching/#dsm):
>
> $$L_{\text{DSM}}(\theta; \sigma) \;=\; \mathbb{E}_{x, \varepsilon}\!\left[\left\|s_\theta(\tilde x, \sigma) + \frac{\varepsilon}{\sigma}\right\|^2\right], \quad \tilde x = x + \sigma \varepsilon$$
>
> These are the same equation if we make two substitutions.

> #### The dictionary
>
> | DDPM | Score Matching |
> |:----:|:--------------:|
> | $x_0$ | $x$ |
> | $x_t$ | $\tilde x$ |
> | $\sqrt{\bar\alpha_t} \, x_0$ | (clean signal — same) |
> | $\sqrt{1 - \bar\alpha_t} \, \epsilon$ | $\sigma \varepsilon$ |
> | $\epsilon_\theta(x_t, t)$ | $-\sqrt{1 - \bar\alpha_t} \, s_\theta(\tilde x, \sigma)$ |
> | $t$ | The noise level (continuous $\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ in some conventions) |
>
> The third row identifies the noise scales: in score matching,
> noise has scale $\sigma$. In DDPM, noise has scale $\sqrt{1 - \bar\alpha_t}$.
> Setting these equal: $\sigma = \sqrt{1 - \bar\alpha_t}$.
>
> The fifth row is the key relationship:
>
> $$\boxed{\;\; s_\theta(x_t, t) \;=\; -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}} \;\;}$$
>
> The DDPM "noise prediction" network is, up to a rescaling factor,
> the **score** of the noise-perturbed distribution
> $q_t = q(x_t \mid x_0)$ marginalized over $x_0$.

> #### Why the conversion factor
>
> The score of $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I)$
> with respect to $x_t$ is (from
> [Gaussian Cookbook §2](../gaussian-cookbook/#mvn-density)):
>
> $$\nabla_{x_t} \log q(x_t \mid x_0) \;=\; -\frac{x_t - \sqrt{\bar\alpha_t} x_0}{1 - \bar\alpha_t} \;=\; -\frac{\sqrt{1 - \bar\alpha_t} \, \epsilon}{1 - \bar\alpha_t} \;=\; -\frac{\epsilon}{\sqrt{1 - \bar\alpha_t}}$$
>
> Predicting $\epsilon$ is, up to a scaling factor of
> $-\sqrt{1 - \bar\alpha_t}$, the same as predicting the score.

> #### What's "annealed" in DDPM sampling
>
> Score Matching used a noise schedule $\sigma_1 > \sigma_2 > \cdots > \sigma_L$
> and ran [annealed Langevin](../score-matching/#annealed-langevin):
> $T$ inner steps at each $\sigma_\ell$.
>
> DDPM runs **one step at each of $T = 1000$ noise levels**, in
> sequence. The "annealing" is implicit in the noise schedule.

> #### Why the variances $\sigma_t$ in the sampler
>
> DDPM Algorithm 2 adds noise at each step: $x_{t-1} = \mu_\theta + \sigma_t z$.
> This noise injection is **exactly** the Langevin diffusion noise
> from [Score Matching §6](../score-matching/#langevin). The two
> framings produce identical update rules — the paper-derived
> $\sigma_t$ (either $\beta_t$ or $\tilde\beta_t$) plays the role of
> $\sqrt{2 \eta}$ in the Langevin update.

> #### The equivalence in one sentence
>
> > **DDPM training is denoising score matching at $T$ noise levels;
> > DDPM sampling is annealed Langevin dynamics.**
>
> This is the punchline of the paper's §3.2. Everything from §2–§6
> of the lesson — the variational derivation, the loss decomposition,
> the $\epsilon$-parameterization — converges here.

**Cross-link callout — back to Score Matching** (`type=back`):

> **Uses: [Score Matching §5 (DSM)](../score-matching/#dsm) and
> [§7 (annealed Langevin)](../score-matching/#annealed-langevin)**.
> The equivalence runs both ways: the score-matching lesson framed
> things in terms of $\sigma$, the DDPM lesson in terms of $t$. The
> conversion is $\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ if
> you treat $\sigma$ as standard-deviation-of-noise-given-clean-data.

**Visualization 8 — `<ScoreEquivalence>`** (full width):

A side-by-side comparison of the two frameworks. Two columns:
- **Left**: DDPM. Animation: $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$,
  the model predicts $\epsilon$, gets compared to the true $\epsilon$.
- **Right**: Score matching. Animation: $\tilde x = x + \sigma \varepsilon$,
  the model predicts the score $s_\theta(\tilde x, \sigma) = -\varepsilon/\sigma$,
  gets compared to the true $-\varepsilon/\sigma$.

A live "conversion bar" between the columns shows the dictionary
substitutions taking left → right and vice versa. As $t$ (DDPM) or
$\sigma$ (SM) changes via a synced slider, both columns update.

Bottom annotation: "Different framings, same training signal. The
DDPM and score-matching gradients on any given $(x_0, \epsilon, t)$
agree up to a per-$t$ rescaling — which is exactly the rescaling
that $L_{\text{simple}}$ chooses to drop."

---

### Section 8 — The Training Algorithm

**Length**: ~700 words. **Anchor: `training-algorithm`**.

**Prose** (verbatim from the paper, then expanded):

> The paper's Algorithm 1 is six lines:
>
> ```
> Algorithm 1 — Training
> 1: repeat
> 2:    x_0 ~ q(x_0)                  // sample a clean data point
> 3:    t ~ Uniform({1, ..., T})       // pick a random timestep
> 4:    epsilon ~ N(0, I)              // sample noise
> 5:    Take gradient descent step on
>          grad_theta || epsilon - epsilon_theta(sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon, t) ||^2
> 6: until converged
> ```
>
> Six lines that took §1–§7 to motivate. Let me unpack them.

> #### Line 2: $x_0 \sim q(x_0)$
>
> Sample a data point from the training distribution. In practice:
> a minibatch of images from the dataset.

> #### Line 3: $t \sim \mathrm{Uniform}(\{1, \ldots, T\})$
>
> **Random timestep per data point.** Why uniform? Because
> $L_{\text{simple}}$ is the average of $L_{0}, L_{1}, \ldots, L_{T-1}$
> over $t$. Sampling $t$ uniformly gives an unbiased Monte Carlo
> estimate of the average. Each gradient step thus optimizes the
> *expected* per-timestep loss, drawn from a uniform distribution
> over timesteps.
>
> Alternative: importance-sample $t$ proportional to the per-timestep
> loss magnitude. The paper finds uniform sampling works well enough
> and is simpler.

> #### Line 4: $\epsilon \sim \mathcal{N}(0, I)$
>
> Sample noise of the same shape as $x_0$. This is the noise that
> will define the forward sample $x_t$ and the prediction target.

> #### Line 5: the gradient step
>
> Compute $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$
> via the **closed-form jump**. Pass through the network to get
> $\epsilon_\theta(x_t, t)$. Compute MSE against the true $\epsilon$.
> Backprop. Step.
>
> Note: $x_0$ and $\epsilon$ enter only through $x_t$ in the loss,
> but the **gradient flows back through both** via standard autograd.
> The model "sees" only the noisy sample; supervision comes from the
> exact $\epsilon$ that produced it.

> #### Tracing one step numerically
>
> Take a 2D point $x_0 = (1.0, -0.5)$, timestep $t = 500$, noise
> $\epsilon = (0.3, -0.7)$. For the paper's schedule:
>
> - $\sqrt{\bar\alpha_{500}} \approx 0.279$
> - $\sqrt{1 - \bar\alpha_{500}} \approx 0.960$
>
> So:
>
> $$x_{500} \;=\; 0.279 \cdot (1.0, -0.5) + 0.960 \cdot (0.3, -0.7) \;\approx\; (0.567, -0.812)$$
>
> (Pre-verified.) The model receives $(0.567, -0.812)$ and the
> integer $t = 500$, and is asked to output $(0.3, -0.7)$. Loss is
> $\|\epsilon - \epsilon_\theta\|^2$.
>
> Over thousands of $(x_0, t, \epsilon)$ triplets, the network learns
> to invert the forward process: given a noisy sample, what noise
> produced it?

**Cross-link callout — sidebar to ELBO/VI** (`type=sidebar`):

> **Compare**: [ELBO/VI §5](../elbo-vi/#vi-algorithm). Standard VI
> optimizes the full ELBO at each step. DDPM optimizes a random
> term — $L_{t-1}$ for random $t$ — at each step. This is a Monte
> Carlo estimate of the sum of $L_{t-1}$ over all $t$. The
> randomization is what makes the per-step gradient computation
> O(1) instead of O(T).

**Visualization 9 — `<TrainingTrace>`** (full width):

A six-panel animated walkthrough of one training step, in the same
style as the VAE §7 visualization:

1. Sample $x_0$ from the dataset (animation: pick a random point).
2. Sample $t$ (animation: dice roll, lands on a value).
3. Sample $\epsilon$ (animation: draw from a 2D Gaussian).
4. Compute $x_t$ via the closed-form jump (animation: linear
   combination with annotated coefficients).
5. Forward through the network → $\epsilon_\theta$ (animation: a
   small architectural diagram with $(x_t, t)$ entering and
   $\epsilon_\theta$ leaving).
6. Compute loss and backprop (animation: gradients flowing back).

Each panel shows concrete numerical values from the §8 numerical
example. A "Step!" button advances to the next training step with
fresh random samples.

---

### Section 9 — The Sampling Algorithm

**Length**: ~900 words. **Anchor: `sampling-algorithm`**.

**Prose** (verbatim from the paper, then expanded):

> The paper's Algorithm 2 is six lines:
>
> ```
> Algorithm 2 — Sampling
> 1: x_T ~ N(0, I)
> 2: for t = T, ..., 1 do
> 3:    z ~ N(0, I) if t > 1, else z = 0
> 4:    x_{t-1} = (1 / sqrt(alpha_t)) (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
> 5: end for
> 6: return x_0
> ```
>
> Sample pure noise. Then for each of $T$ steps, apply the reverse
> update: compute the conditional mean, add a fresh dose of noise,
> repeat. After $T$ steps, you have a sample from $p_\theta(x_0)$.

> #### Reading the update rule
>
> $$x_{t-1} \;=\; \underbrace{\frac{1}{\sqrt{\alpha_t}}\!\left(x_t \;-\; \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}} \, \epsilon_\theta(x_t, t)\right)}_{\mu_\theta(x_t, t)} \;+\; \sigma_t \, z$$
>
> The first part is the $\epsilon$-parameterized $\mu_\theta(x_t, t)$
> from §6. The second part — adding $\sigma_t z$ — is **Langevin
> noise injection** from [Score Matching §6](../score-matching/#langevin).
>
> Why add noise during sampling? Two reasons:
>
> 1. **It's variational inference.** The reverse process $p_\theta(x_{t-1} \mid x_t)$
>    is parameterized as a Gaussian; sampling means drawing from
>    that Gaussian (mean plus noise). Setting $z = 0$ would collapse
>    every sample to the mean — not actually sampling.
> 2. **It's Langevin sampling.** Pure gradient descent on
>    $\log p$ collapses to modes. The noise term is what lets the
>    sampler explore and approximate the full distribution.
>
> The two reasons are the same reason in different framings. (Of
> course they are — that's the whole §7 punchline.)

> #### The final step ($t = 1$)
>
> At $t = 1$, we set $z = 0$. The final sample is $\mu_\theta(x_1, 1)$
> — deterministic given $x_1$. Why? Because $x_0$ should be a clean
> data point, not have additional Gaussian noise added at the end.
> The "Langevin temperature" cools to zero at the final step.
>
> Practically: sampling $x_0 \sim p_\theta(x_0 \mid x_1)$ with the
> discretized Gaussian decoder (Eq 13 of the paper) for image data;
> for continuous data, $x_0 = \mu_\theta(x_1, 1)$ directly.

> #### Tracing one sampling step numerically
>
> Suppose at $t = 500$ we have $x_t = (0.567, -0.812)$ and the model
> outputs $\epsilon_\theta(x_t, 500) = (0.3, -0.7)$ (a "perfect"
> prediction for the §8 trace). Fresh noise $z = (0.1, -0.2)$,
> $\sigma_t = \sqrt{\beta_{500}} \approx 0.1003$:
>
> - $\mu_\theta = \frac{1}{\sqrt{0.9899}}\!\left((0.567, -0.812) - \frac{0.0101}{0.9603}(0.3, -0.7)\right)$
> - $\quad\;\;\, \approx (0.5667, -0.8084)$
> - $\sigma_t z = (0.01003, -0.02006)$
> - $x_{t-1} \approx (0.577, -0.828)$
>
> The sample has moved very slightly toward $x_0$ (which the model
> would have inferred to be $(1.0, -0.5)$ via the
> $x_0 = (x_t - \sqrt{1 - \bar\alpha_t} \epsilon_\theta) / \sqrt{\bar\alpha_t}$
> formula). Multiplied over 500 more steps, the sample lands in
> the data distribution.
>
> (Pre-verified; test target in `reverse-process.test.ts`.)

> #### What $\hat x_0$ means
>
> At any timestep during sampling, we can compute the model's
> "running estimate of the clean data":
>
> $$\hat x_0(x_t, t) \;=\; \frac{1}{\sqrt{\bar\alpha_t}}\!\left(x_t \;-\; \sqrt{1 - \bar\alpha_t} \, \epsilon_\theta(x_t, t)\right)$$
>
> This is what the model "thinks" $x_0$ is, given the current noisy
> sample and predicted noise.
>
> $\hat x_0(x_T, T)$ — at the start of sampling, with $x_T$ pure
> noise — is a noisy guess. $\hat x_0(x_1, 1)$ — at the end — is a
> sharp estimate. Watching the **evolution of $\hat x_0$** through
> sampling is one of the most pedagogically illuminating things you
> can do; it's how the paper's Figure 6 (CIFAR10 progressive
> generation) is computed.

**Visualization 10 — `<SamplingChain>`** (full width):

A 2D canvas. On "Start sampling":
- 100 particles initialized at $\mathcal{N}(0, I)$.
- Each particle steps through $t = T, T-1, \ldots, 1$ via the
  Algorithm 2 update, using the **trained** $\epsilon_\theta(x_t, t)$.
- The canvas animates the particles' positions over time.
- A timestep counter shows the current $t$.

After 100 steps (we use $T = 100$ for browser speed): particles
cluster at the four data modes, distributed roughly evenly. The
animation is the "noise turns into data" payoff.

Two toggles:
- **Show $\hat x_0$ overlay**: at each frame, also plot each
  particle's $\hat x_0(x_t, t)$ as a translucent ghost dot. Early in
  sampling, $\hat x_0$ is very noisy (low-confidence guess); late in
  sampling, it converges to the actual sample. Watch the ghost
  positions sharpen.
- **Show score field overlay**: at the current $t$, overlay the
  learned score $s_\theta = -\epsilon_\theta / \sqrt{1 - \bar\alpha_t}$
  as a quiver plot. Watch the field morph from smooth-and-global
  at large $t$ to sharp-and-modal at small $t$.

Buttons: Play / Pause / Step / Reset.

This visualization is the **sampling animation** of the lesson. §10's
centerpiece extends it.

---

### Section 10 — A Trained 2D DDPM You Can Explore

**Length**: ~1100 words. **The centerpiece.**

**Prose**:

> The §9 sampling chain is one mode of interaction. This section
> turns the trained 2D DDPM into a **full explorer**: forward
> diffusion, reverse sampling, $\hat x_0$ trajectories, interpolation
> in latent space (recreating Figures 8 and 9 of the paper), and
> direct comparison with VAE and Score Matching outputs.
>
> The model: 2D synthetic data (the same 4-cluster mixture used
> throughout the curriculum), $T = 100$ timesteps (small enough for
> 30 fps animation), $\epsilon_\theta$ implemented as an MLP that
> takes $(x_t, t)$ where $t$ is sinusoidally embedded. ~5000
> parameters total. Pre-trained offline; weights ship as JSON.
>
> #### What the explorer lets you do
>
> **Mode 1 — Forward chain.** Pick a data point. Watch it diffuse
> over $T$ steps into noise. Slider for $t$ inspects any intermediate
> timestep. Overlay: the closed-form marginal $q(x_t \mid x_0)$ as
> an ellipse. Verifies §2.
>
> **Mode 2 — Reverse sampling.** Pick a noise sample (or click "Re-roll").
> Animate the reverse process. Watch the sample evolve from pure
> noise to a data-distribution sample. Verifies §9.
>
> **Mode 3 — $\hat x_0$ evolution.** Run a reverse-sampling chain
> and visualize the model's $\hat x_0$ estimate at every step.
> Early estimates are noisy (low $\bar\alpha_t$, high uncertainty);
> later estimates sharpen. The $\hat x_0$ trajectory traces a path
> from "a generic guess at the data center" to a specific cluster.
> Compare to the paper's Figure 6.
>
> **Mode 4 — Latent interpolation (Figure 8 recreation).** Pick two
> data points $x_0, x_0'$. The explorer diffuses both to timestep
> $t^*$ (slider-controlled), producing $x_{t^*}$ and $x'_{t^*}$.
> Linearly interpolate: $\bar x_{t^*}(\lambda) = (1-\lambda) x_{t^*} + \lambda x'_{t^*}$
> for $\lambda \in [0, 1]$. Run the reverse process from $\bar x_{t^*}$
> to get $\bar x_0(\lambda)$. Plot all the $\bar x_0$ outputs for ten
> values of $\lambda$.
>
> At small $t^*$: interpolations stay near the source images
> (limited mixing). At large $t^*$: interpolations are essentially
> independent samples (full mixing). At intermediate $t^*$: smooth
> morphing through latent space. **This recreates the paper's
> coarse-to-fine interpolation phenomenon.**
>
> **Mode 5 — Score field overlay.** At any $t$ during reverse
> sampling, overlay the learned score field
> $s_\theta = -\epsilon_\theta / \sqrt{1 - \bar\alpha_t}$. **This
> directly visualizes the §7 equivalence**: DDPM's $\epsilon$ network,
> rescaled, is the score field of the noise-smoothed distribution.
> Toggle on/off; the field morphs from smooth-and-global (large $t$)
> to sharp-and-modal (small $t$).
>
> **Mode 6 — Cross-model comparison.** Same data, three trained
> models (DDPM, VAE, Score Matching). Side-by-side panels show:
> - The data distribution (faded background).
> - Samples produced by each model after a full sampling run.
> - The latent / score structure each model has learned.
>
> This is the **synthesis viz** of the curriculum: all three
> generative paradigms on the same problem, with quality
> measurements (sample diversity, mode coverage, density estimation
> quality).

**Visualization 11 — `<TrainedDDPMExplorer>`** (full width, ≥900px tall).

The centerpiece. Layout:

```
┌────────────────────────────────────────────────────────────────────┐
│  MODE SELECTOR (tabs)                                                │
│  [ Forward ] [ Reverse ] [ x̂_0 ] [ Interpolation ] [ Score ] [ Cmp ] │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                       MAIN CANVAS (responsive to mode)               │
│                                                                      │
│                                                                      │
├────────────────────────────────────────────────────────────────────┤
│  TIMESTEP SLIDER:  t = 0 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ T │
│  CONTROLS: [Play] [Pause] [Step] [Reset]   Speed: 0.5x ●━━━ 5x      │
│  MODE-SPECIFIC OPTIONS appear below depending on selected mode      │
└────────────────────────────────────────────────────────────────────┘
```

Per-mode details:

**Mode 1 (Forward)**: select a starting $x_0$ by clicking on the
canvas. Animation shows $x_t$ along its forward trajectory. The
1000-point training set is shown faded in the background; the
selected $x_0$ is highlighted, and its $x_t$ at the current slider
position is rendered with the $q(x_t \mid x_0)$ ellipse overlaid.

**Mode 2 (Reverse)**: click "Re-roll" to sample a fresh $x_T$.
Animation runs the reverse chain, showing $x_t$ at each step. Trail
of recent positions in `--reverse-arrow` opacity-faded.

**Mode 3 ($\hat x_0$)**: like Mode 2, but also shows $\hat x_0(x_t, t)$
at each step. The $\hat x_0$ trajectory traces from "generic
center" to specific cluster.

**Mode 4 (Interpolation)**: click two data points to mark them as
$x_0$ and $x_0'$. Slider for $t^*$. Below the canvas: a row of 10
reconstructed $\bar x_0$ outputs for $\lambda = 0, 0.1, 0.2, \ldots, 1.0$.
This is the 2D analog of the paper's Figure 9.

**Mode 5 (Score)**: overlay quiver field of learned score on the
current canvas. Color-coded arrows in `--score-vec`. Updates as the
timestep slider moves.

**Mode 6 (Comparison)**: three small panels — DDPM, VAE, Score
Matching — each showing samples and the data distribution.
Below: a small table with metrics (FID-like distance to data
distribution, sample diversity, etc.). Read in:
- VAE: sharp clustering but some interpolation between modes
  (typical of VAE: smooth latent space).
- Score Matching: spread out samples but mode coverage depends on
  noise schedule details.
- DDPM: best mode coverage and sample quality.

This visualization is **the deliverable of the entire curriculum**.
It should be polished, smooth, instantly responsive.

**Visualization 12 — `<CoarseToFineInterp>`** (full width):

A focused recreation of Figure 9 from the paper (which shows
CelebA-HQ interpolations at varying diffusion depths) but for 2D
data. A grid:
- Rows: $t^* \in \{0, 25, 50, 75, 100\}$.
- Columns: $\lambda \in \{0, 0.1, \ldots, 1.0\}$.

Each cell renders the $\bar x_0(\lambda)$ output. Patterns:
- Row $t^* = 0$: cells are linear interpolations in data space
  (cross the empty between-cluster region).
- Row $t^* = T$: cells are independent samples (no relation to the
  source points).
- Middle rows: smooth morphing through latent space.

This visualization complements Mode 4 of the main explorer by
showing the **dependence on $t^*$** in a single picture.

---

### Section 11 — Practical Considerations

**Length**: ~700 words.

**Prose**:

> Several design choices in the paper deserve discussion: the noise
> schedule, the variance parameterization $\Sigma_\theta$, the loss
> weighting (full $L$ vs $L_{\text{simple}}$), and the practical
> trade-offs between sample quality and log likelihood.

> #### Noise schedule choice
>
> Paper uses **linear $\beta_t$ from $10^{-4}$ to $0.02$ over $T = 1000$**.
> Considerations:
>
> - **Endpoint $L_T \approx 10^{-5}$ bits/dim**. The signal is
>   destroyed; $q(x_T \mid x_0)$ is essentially $\mathcal{N}(0, I)$.
>   Verified above.
> - **Small $\beta_t$ keeps the local Gaussian approximation valid**.
>   Reverse transitions are well-modeled as Gaussian only when the
>   forward step is small.
> - **Linear is empirically fine but not optimal**. Cosine schedules
>   (Nichol & Dhariwal 2021) and others improve quality on harder
>   datasets. The lesson briefly demonstrates schedule effects via
>   the `<NoiseScheduleExplorer>` visualization.

> #### $\Sigma_\theta$: learned vs fixed
>
> The paper's ablation (Table 2): fixing $\Sigma_\theta = \sigma_t^2 I$
> with $\sigma_t^2 \in \{\beta_t, \tilde\beta_t\}$ works **better**
> than learning a diagonal $\Sigma_\theta(x_t, t)$. Learning the
> variance leads to instability.
>
> Why? Empirically: the variance learning interacts badly with the
> loss-weight-flattening of $L_{\text{simple}}$. Better-quality
> theoretical treatments (e.g., Nichol & Dhariwal's IDDPM, 2021)
> learn an *interpolation coefficient* between $\beta_t$ and
> $\tilde\beta_t$ — a single scalar per $t$ — and recover the
> stability while gaining flexibility.

> #### $L$ vs $L_{\text{simple}}$: the trade-off
>
> Paper's Table 1: training on the full $L$ gives better
> **codelengths** (negative log likelihood); training on
> $L_{\text{simple}}$ gives better **sample quality** (FID).
>
> Why the divergence?
>
> - **Full $L$** matches the variational objective exactly. The
>   model is optimally trained as a density estimator.
> - **$L_{\text{simple}}$** drops the per-timestep weights, which
>   re-balances the training signal. The model spends more capacity
>   on harder (large-$t$) denoising tasks. This is suboptimal for
>   density estimation but better for the perceptual quality of
>   samples.
>
> The choice depends on the application. If you want a compressor
> that achieves good bits/dim: full $L$. If you want pretty pictures:
> $L_{\text{simple}}$.

> #### Progressive lossy compression
>
> The paper's §4.3 reframes the trained model as a **progressive
> lossy compressor**. Algorithms 3 and 4 in the paper transmit a
> sample using ~$D_{\mathrm{KL}}(q(x) \mid p(x))$ bits, with the
> receiver progressively decoding from coarse (large $t$) to fine
> (small $t$). The rate-distortion curve (Figure 5 of the paper):
> most of the bits are spent on imperceptible details.
>
> This isn't a deployable compressor (it requires minimal random
> coding, which isn't tractable), but it's a useful conceptual
> framing. **Diffusion models are progressive decoders**, and this
> structure is what makes them so good at generating coherent global
> structure first and fine details last.

> #### Variants and extensions (sketched, links out)
>
> - **DDIM** (Song et al. 2021): same trained network, deterministic
>   sampler ($\sigma_t = 0$). 10× faster sampling at slightly lower
>   quality. Most practical implementations use DDIM-style sampling.
> - **Classifier guidance** (Dhariwal & Nichol 2021): condition the
>   sampling on a class label by adding a classifier's gradient at
>   each step.
> - **Classifier-free guidance** (Ho & Salimans 2021): train a
>   conditional and unconditional model jointly; combine at sampling
>   for tunable conditional strength. The basis of all modern
>   conditional diffusion (Stable Diffusion, Imagen, etc.).
> - **Latent diffusion** (Rombach et al. 2022): run the diffusion in
>   the latent space of a pre-trained autoencoder. The basis of
>   Stable Diffusion.
> - **Score-based SDEs** (Song et al. 2021): the continuous-time
>   limit. Reformulates DDPM as the discretization of a stochastic
>   differential equation. Beautiful and powerful framework.

**Visualization 13 — `<NoiseScheduleExplorer>`** (medium width):

Slider for "schedule type": linear, cosine, sigmoid. As the user
selects:
- Top plot: $\beta_t$ vs $t$ for each schedule.
- Bottom plot: $\sqrt{\bar\alpha_t}$ (signal strength) vs $t$.

Annotations show the differences:
- Linear (paper's choice): $\bar\alpha_t$ decays roughly geometrically.
  Most signal destruction in the middle of the schedule.
- Cosine (Nichol & Dhariwal): $\bar\alpha_t$ decays slowly at first
  and last, fast in the middle. More uniform information destruction.
- Sigmoid: similar to cosine but with adjustable midpoint.

Pedagogical point: "the schedule controls **when** information is
destroyed. Better schedules destroy information more uniformly."

---

### Section 12 — Where You'll See This (and What's Next)

**Length**: ~500 words.

**Prose**:

> You've reached the destination of StatViz.
>
> Diffusion models are, as of 2024, the dominant generative modeling
> paradigm for image, video, audio, and increasingly other modalities.
> The DDPM paper you've just read in full is the foundation. Modern
> systems extend it in several directions, but the core machinery is
> what's in this lesson.

> #### Direct descendants
>
> - **Stable Diffusion** (Rombach et al. 2022): DDPM in the latent
>   space of a pre-trained VAE. Faster, more efficient, conditionable.
> - **Imagen** (Saharia et al. 2022) and **DALL-E 2** (Ramesh et al.
>   2022): text-to-image diffusion at scale. Classifier-free guidance
>   on a text-conditioned $\epsilon_\theta(x_t, t, c)$.
> - **Video diffusion** (Ho et al. 2022): factorize space and time;
>   diffusion over a 3D tensor. Sora and friends.
> - **Audio diffusion**: AudioLM, Riffusion, Suno. Direct application
>   to 1D signals.

> #### Theoretical/mathematical extensions
>
> - **Score-based SDEs** (Song et al. 2021): continuous-time limit;
>   the forward process is an SDE, the reverse is its time-reversed
>   SDE. The discrete DDPM is one possible discretization. Other
>   discretizations (DDIM, DPM-Solver) yield faster samplers.
> - **Flow matching** (Lipman et al. 2023): an ODE-based alternative
>   to diffusion. Same goal (transport noise to data), different
>   mathematical machinery. Often easier to train.
> - **Schrödinger bridges** (De Bortoli et al. 2021): the most general
>   formulation, of which DDPM is a special case.
> - **Consistency models** (Song et al. 2023): few-step (sometimes
>   single-step) sampling. Distillation from a trained diffusion
>   model.

> #### Where you can go from here
>
> You're now equipped to read essentially any modern generative
> modeling paper. The natural follow-ons:
>
> 1. **Score-based SDEs** (Song et al. 2021) — continuous-time
>    generalization. Builds directly on this lesson + the Score
>    Matching lesson.
> 2. **Latent diffusion** (Rombach et al. 2022) — combines VAE and
>    DDPM. Direct combination of two lessons you've read.
> 3. **Classifier-free guidance** (Ho & Salimans 2021) — a few-page
>    paper, easy read.
> 4. **Flow matching** (Lipman et al. 2023) — an alternative to
>    diffusion using ODEs. Builds on Score Matching directly.
> 5. **Sidebars** mentioned earlier in the curriculum:
>    Normalizing Flows, MCMC, Langevin dynamics (standalone). All
>    now reachable.

**Visualization 14 — `<FinalRoadmap>`** (full width):

The complete StatViz roadmap, with **all seven lessons lit**: EM,
KL & Jensen, ELBO/VI, Gaussian Cookbook, VAE, Score Matching, DDPM.
The "golden thread" highlighted as a complete path from KL → ELBO →
Cookbook → VAE & Score Matching → DDPM. Sidebar lessons (MCMC,
Normalizing Flows, Langevin) shown dimmed as future expansions.

A final annotation: **"You've finished StatViz. The DDPM paper is
now readable to you in its entirety. Go read it again — it'll feel
very different this time."**

---

## 5. Algorithm / Math Implementation

### `src/lessons/ddpm/math/schedule.ts`

```ts
/** Linear noise schedule from beta_start to beta_end over T steps. */
export interface Schedule {
  T: number;
  betas: number[];        // beta_t
  alphas: number[];       // 1 - beta_t
  alpha_bars: number[];   // cumprod
  alpha_bars_prev: number[]; // [1, alpha_bar_1, ..., alpha_bar_{T-1}]
  tilde_betas: number[];  // (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
}

export function linearSchedule(T: number, beta_start: number, beta_end: number): Schedule {
  const betas = Array.from({ length: T }, (_, i) =>
    beta_start + (beta_end - beta_start) * i / (T - 1)
  );
  const alphas = betas.map(b => 1 - b);
  const alpha_bars: number[] = [];
  let cum = 1;
  for (const a of alphas) { cum *= a; alpha_bars.push(cum); }
  const alpha_bars_prev = [1, ...alpha_bars.slice(0, -1)];
  const tilde_betas = betas.map((b, t) =>
    (1 - alpha_bars_prev[t]) / (1 - alpha_bars[t]) * b
  );
  return { T, betas, alphas, alpha_bars, alpha_bars_prev, tilde_betas };
}
```

### `src/lessons/ddpm/math/forward-process.ts`

```ts
import type { Schedule } from './schedule';

/** Sample x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps. */
export function forwardSample(x_0: number[], t: number, eps: number[], sched: Schedule): number[] {
  const sqrtA = Math.sqrt(sched.alpha_bars[t]);
  const sqrt1mA = Math.sqrt(1 - sched.alpha_bars[t]);
  return x_0.map((x, i) => sqrtA * x + sqrt1mA * eps[i]);
}

/** Forward posterior mean tilde_mu_t(x_t, x_0). */
export function posteriorMean(x_t: number[], x_0: number[], t: number, sched: Schedule): number[] {
  const ab = sched.alpha_bars[t];
  const ab_prev = sched.alpha_bars_prev[t];
  const at = sched.alphas[t];
  const bt = sched.betas[t];
  const coef0 = Math.sqrt(ab_prev) * bt / (1 - ab);
  const coefT = Math.sqrt(at) * (1 - ab_prev) / (1 - ab);
  return x_0.map((x0i, i) => coef0 * x0i + coefT * x_t[i]);
}

/** Forward posterior, computed from x_t and the (true or predicted) epsilon. */
export function posteriorMeanFromEps(x_t: number[], eps: number[], t: number, sched: Schedule): number[] {
  const at = sched.alphas[t];
  const ab = sched.alpha_bars[t];
  const coef = sched.betas[t] / Math.sqrt(1 - ab);
  const factor = 1 / Math.sqrt(at);
  return x_t.map((xi, i) => factor * (xi - coef * eps[i]));
}

/** Recover x_hat_0 from x_t and predicted epsilon. */
export function xHat0(x_t: number[], eps: number[], t: number, sched: Schedule): number[] {
  const sqrtA = Math.sqrt(sched.alpha_bars[t]);
  const sqrt1mA = Math.sqrt(1 - sched.alpha_bars[t]);
  return x_t.map((xi, i) => (xi - sqrt1mA * eps[i]) / sqrtA);
}
```

### `src/lessons/ddpm/math/reverse-process.ts`

```ts
import type { Schedule } from './schedule';
import { posteriorMeanFromEps } from './forward-process';

/** One Algorithm 2 reverse step. */
export function reverseStep(
  x_t: number[],
  eps_pred: number[],
  t: number,
  z: number[],
  sched: Schedule,
  sigma_choice: 'beta' | 'tilde_beta' = 'beta',
): number[] {
  const mu = posteriorMeanFromEps(x_t, eps_pred, t, sched);
  if (t === 0) return mu; // no noise on final step
  const sigma2 = sigma_choice === 'beta' ? sched.betas[t] : sched.tilde_betas[t];
  const sigma = Math.sqrt(sigma2);
  return mu.map((m, i) => m + sigma * z[i]);
}
```

### `src/lessons/ddpm/math/score-conversion.ts`

```ts
import type { Schedule } from './schedule';

/** Convert epsilon prediction to score: s_theta = -eps_theta / sqrt(1 - alpha_bar_t). */
export function epsToScore(eps_pred: number[], t: number, sched: Schedule): number[] {
  const denom = Math.sqrt(1 - sched.alpha_bars[t]);
  return eps_pred.map(e => -e / denom);
}

/** Inverse: s -> eps_theta = -sqrt(1 - alpha_bar_t) * s. */
export function scoreToEps(score: number[], t: number, sched: Schedule): number[] {
  const factor = Math.sqrt(1 - sched.alpha_bars[t]);
  return score.map(s => -factor * s);
}
```

### Test cases

- Schedule check: `alpha_bars[500]` ≈ 0.0778, `alpha_bars[999]` ≈ $4 \times 10^{-5}$
- Forward sample: with $x_0 = (1.0, -0.5)$, $\epsilon = (0.3, -0.7)$,
  $t = 500$: $x_t \approx (0.567, -0.812)$.
- Posterior mean equivalence: `posteriorMean(x_t, x_0, t, sched)` equals
  `posteriorMeanFromEps(x_t, epsilon_true, t, sched)` for the true noise.
- Recover $x_0$: `xHat0(x_t, true_eps, t, sched)` equals the original $x_0$.
- Reverse step: with perfect $\epsilon$ prediction and $z = 0$,
  $x_{t-1} = $ posterior mean.
- Score conversion roundtrip: `scoreToEps(epsToScore(eps, t, sched), t, sched)` = `eps`.
- $L_T$ check: for the paper's schedule, $D_{\mathrm{KL}}(\mathcal{N}(\sqrt{\bar\alpha_T} x_0, (1-\bar\alpha_T) I) \,\|\, \mathcal{N}(0, I))$
  for unit-norm $x_0$ is ≈ $2 \times 10^{-5}$ nats per dim, ≈ $3 \times 10^{-5}$ bits per dim.

---

## 6. Component Catalog

### Shared (already exist)
Standard chrome from `@shared/ui/`.

### Lesson-local — 14 visualizations
- `<HeroAnimation>` (§1)
- `<ForwardChainViewer>` (§2)
- `<ClosedFormJump>` (§2)
- `<GraphicalModel>` (§3)
- `<VLBDecomposition>` (§4)
- `<ForwardPosteriorExplorer>` (§5)
- `<ParameterizationComparison>` (§6)
- `<ScoreEquivalence>` (§7)
- `<TrainingTrace>` (§8)
- `<SamplingChain>` (§9)
- `<TrainedDDPMExplorer>` (§10) — **the centerpiece**
- `<CoarseToFineInterp>` (§10)
- `<NoiseScheduleExplorer>` (§11)
- `<FinalRoadmap>` (§12)

---

## 7. Page-Level UX

Same as other lessons. `<PrereqStrip>` shows five required prereqs
(EM, KL, ELBO, Cookbook, VAE, Score Matching) — the most dense prereq
strip in the curriculum. Notes specific to this lesson:

1. **The §10 explorer is the polish-budget anchor.** Every other
   visualization should be functional and good; §10 must be
   *exceptional*. Smooth animations, instant interactions, clear
   mode-switching. This is the artifact a learner remembers.

2. **Long-form support.** The lesson is ~3-4 hours. Provide a
   sticky table-of-contents that scrolls with the user, so they can
   navigate without losing their place. Mark visited sections.

3. **The "final celebration" page.** After §12, render a small "you
   made it" page with the full roadmap lit, a link back to the
   curriculum index, and an invitation to revisit the DDPM paper.

---

## 8. Acceptance Criteria

A learner who has worked through this page should be able to, on a
blank sheet:

1. State the forward process definition and the closed-form
   $q(x_t \mid x_0)$ marginal.
2. State the variational lower bound and identify the three terms
   $L_T, L_{t-1}, L_0$.
3. Derive the forward posterior $q(x_{t-1} \mid x_t, x_0)$ from
   first principles (with help from the Gaussian Cookbook).
4. Derive the $\epsilon$-prediction parameterization and explain
   why it produces stable, uniform-scale training targets.
5. State $L_{\text{simple}}$ and explain how dropping the
   per-timestep weighting affects training.
6. State Algorithm 1 and Algorithm 2 from memory, with each line
   annotated.
7. Explain — in two sentences each — why DDPM training is denoising
   score matching and why DDPM sampling is annealed Langevin.
8. Read a paragraph from the DDPM paper out loud and translate it
   into mechanical operations they could implement.
9. Identify what would change in the algorithm under the variants
   discussed in §11.

---

## 9. Stretch Goals (post-MVP)

- **In-browser training**: a "Train this DDPM" button that runs
  ~200 gradient steps in the browser on the 2D toy. Heavy lift but
  the ultimate payoff.
- **Tiny image DDPM**: a 10×10 grayscale image DDPM trained on MNIST.
  Ship pre-trained, animate sampling. Closer to the paper's actual
  use case.
- **DDIM toggle in §10**: add a "Deterministic (DDIM)" mode that
  runs sampling with $\sigma_t = 0$. Show that samples are still
  reasonable but the chain is deterministic given $x_T$. Trade-off:
  faster but less diverse.
- **The reverse-time SDE callout**: a sidebar in §7 introducing the
  continuous-time formulation (Song 2021). Sets up a future
  "Score-based SDEs" sidebar lesson.
- **The classifier-free guidance demo**: a fourth panel in §10
  showing conditional generation on a "select the cluster" toggle.

---

## 10. Out of Scope (intentionally)

- **The U-Net architecture details** (Appendix B of the paper). The
  lesson's 2D MLP is sufficient to demonstrate everything; image-scale
  architectural choices belong in a paper-focused tutorial.
- **The discretized Gaussian decoder for $L_0$** (Eq 13 of the paper).
  Briefly mentioned in §4; not derived. Image-data-specific.
- **The progressive lossy compression algorithm** (Algorithms 3
  and 4 of the paper). Briefly mentioned in §11; not implemented.
- **Connection to autoregressive decoding** (§4.3 of the paper).
  Brief mention in §12; full treatment is a sidebar topic.
- **Importance-weighted DSM weighting schemes** (Karras et al. 2022,
  EDM). Mentioned in §11; not derived.
- **Continuous-time SDE limit** — mentioned as a follow-on in §12,
  not developed here.

---

## 11. Training Notebook (offline pre-step)

The lesson ships pre-trained weights for the §10 trained DDPM. The
notebook (`docs/ddpm-training-notebook.ipynb`) should:

```python
# Pseudocode — agent expands and runs this
import numpy as np, torch, torch.nn as nn, json

# 1. 4-cluster 2D dataset (consistency with VAE, Score Matching)
np.random.seed(0)
centers = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2]])
N_per_cluster = 250
X = np.vstack([np.random.normal(c, 0.2, (N_per_cluster, 2)) for c in centers]).astype(np.float32)

# 2. Noise schedule (T = 100 for browser speed)
T = 100
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# 3. Epsilon network
class EpsNet(nn.Module):
    def __init__(self, hidden=64, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        # Sinusoidal embedding -> MLP
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
    def time_embed(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half) / (half - 1))
        emb = t[:, None] * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
    def forward(self, x, t):
        emb = self.time_embed(t.float())
        return self.net(torch.cat([x, emb], dim=-1))

# 4. Lsimple training
model = EpsNet(hidden=64, time_dim=32)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xt = torch.tensor(X)
for epoch in range(20000):
    idx = np.random.choice(len(Xt), 128, replace=False)
    x_0 = Xt[idx]
    t = torch.randint(0, T, (128,))
    eps = torch.randn_like(x_0)
    ab_t = alpha_bars[t].unsqueeze(-1)
    x_t = torch.sqrt(ab_t) * x_0 + torch.sqrt(1 - ab_t) * eps
    eps_pred = model(x_t, t)
    loss = ((eps - eps_pred) ** 2).sum(dim=-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 5. Save weights + schedule
weights = {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}
weights['_metadata'] = {
    'T': T,
    'betas': betas.tolist(),
    'alpha_bars': alpha_bars.tolist(),
    'data_centers': centers.tolist(),
    'hidden_dim': 64,
    'time_dim': 32,
    'epochs': 20000,
}
with open('src/lessons/ddpm/assets/ddpm-weights.json', 'w') as f:
    json.dump(weights, f)
```

After training, **validate offline**:
1. Sample 1000 reverse-process trajectories. Plot endpoints.
   They should cluster at the four data modes with roughly equal
   counts.
2. Plot the learned score field $-\epsilon_\theta(x, t) / \sqrt{1 - \bar\alpha_t}$
   at $t \in \{1, 25, 50, 75, 99\}$. At small $t$, should match the
   analytical score of the data distribution (use the score-matching
   lesson's `scoreSmoothedGMM` for ground truth). At large $t$,
   should match the score of $\mathcal{N}(0, I)$ (i.e., $s(x) = -x$).
3. Trace $\hat x_0(x_t, t)$ for several reverse-sampling trajectories.
   Should converge to a specific data point as $t \to 0$.

If validation fails (samples clustered in only 2 of 4 clusters, or
the score field at small $t$ misses the modes), retrain with more
epochs or a wider network.