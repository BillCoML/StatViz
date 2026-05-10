# ELBO & Variational Inference — Interactive Lesson
## Build Specification & Content Plan

> A single-page, deeply explained, interactive lesson on the Evidence
> Lower Bound (ELBO) and variational inference. The lesson establishes
> the **fundamental identity** $\log p(x) = \mathrm{ELBO}(q) + D_{\mathrm{KL}}(q \,\|\, p_{\text{posterior}})$
> and shows that maximizing ELBO over $q$ is equivalent to minimizing
> reverse KL to the posterior. Includes a striking re-derivation of EM
> as coordinate ascent on the ELBO.
>
> **Position in the roadmap**: foundational. Required for VAE and DDPM.
> Builds directly on the KL & Jensen lesson (uses Jensen's inequality
> in the derivation, uses Gibbs' inequality and the forward/reverse-KL
> dichotomy as conceptual scaffolding). Backfills a missing perspective
> on the EM lesson — the EM convergence proof is, in modern language,
> coordinate ascent on this objective.

---

## 0. Pedagogical Philosophy

Same commitments as the EM and KL lessons:

1. **Concrete before abstract.** A latent-variable picture (observed
   $x$, latent $z$, intractable posterior) is established before any
   ELBO formula appears.
2. **Math is shown in full.** Both derivations of the ELBO (Jensen
   route + KL identity route). Full derivation of the
   reconstruction-KL form. Full computation of the worked Gaussian
   example. Full demonstration that EM is coordinate ascent on ELBO.
3. **Visualization carries half the load.** The "ELBO + KL = log p(x)"
   identity gets a centerpiece visualization where the reader watches
   the bound tighten as $q$ approaches the posterior.
4. **Cross-page integration is essential.** The bimodal posterior in
   the worked example **is** the bimodal target from KL §7 — the
   reverse-KL fit there is the same fit VI produces here. The EM
   re-derivation **is** the EM lesson, told with new vocabulary.

By the end, the reader can: (a) state and prove the fundamental
identity, (b) write both forms of the ELBO and explain when each is
useful, (c) explain why VI is "approximate EM" and when the
approximation is exact, (d) predict (qualitatively) the behavior of VI
on a misspecified family.

---

## 1. Tech Stack

Identical to KL and EM lessons. Vite + TypeScript multi-page (under the
StatViz monorepo described in `system_roadmap.md`), KaTeX, D3, hand-
written CSS using shared tokens, Fraunces / Source Serif 4 / JetBrains
Mono.

File layout under the StatViz monorepo (per `system_roadmap.md` §3):

```
src/lessons/elbo-vi/
├── main.ts
├── meta.ts                          # lesson metadata
├── math/
│   ├── gaussian.ts                  # exact Gaussian-conjugate computations
│   ├── bimodal.ts                   # numerical posterior + ELBO grid
│   ├── em-elbo.ts                   # 2-coins ELBO at (q, theta)
│   └── *.test.ts
├── sections/
│   ├── 01-hook.ts
│   ├── 02-the-setup.ts
│   ├── 03-elbo-identity.ts
│   ├── 04-two-forms.ts
│   ├── 05-vi-algorithm.ts
│   ├── 06-worked-examples.ts
│   ├── 07-em-as-elbo-ascent.ts
│   └── 08-where-youll-see-this.ts
├── viz/
│   ├── elbo-kl-decomposition.ts     # §3 — the fundamental identity
│   ├── elbo-optimization.ts         # §6 Act 1
│   ├── bimodal-elbo.ts              # §6 Act 2 — centerpiece
│   ├── em-trajectory.ts             # §7 — ELBO during EM iterations
│   └── posterior-comparison.ts      # §6 sub-component
└── styles/
    └── overrides.css                # lesson-local accents only
```

The shared chrome (PrereqStrip, CrosslinkCallout, RoadmapMini, etc.)
comes from `@shared/ui`. Math primitives that already exist in other
lessons (e.g., `klGaussian` from `@lessons/kl-jensen/math/kl`) are
imported directly.

---

## 2. Visual / Aesthetic Direction

Same paper-and-ink aesthetic. Lesson-local accents:

```css
--evidence:    #6b3a8c;   /* log p(x) — deep purple, the "truth" */
--elbo:        #2c5f8d;   /* ELBO — slate blue, same as q in KL */
--kl-gap:      #b8651a;   /* KL gap — burnt sienna, same as p in KL */
--posterior:   #3a7058;   /* the true posterior — forest green */
--variational: #c87f3b;   /* the variational q — ochre */
```

The `--posterior` and `--variational` colors are used consistently for
the two distributions throughout §6 and §7. The `--evidence`,
`--elbo`, `--kl-gap` colors are used for the three quantities in the
fundamental identity in §3.

---

## 3. Lesson Metadata (`src/lessons/elbo-vi/meta.ts`)

```ts
import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'elbo-vi',
  title: 'ELBO & Variational Inference',
  subtitle: 'Lower bounds, variational families, and why EM is just coordinate ascent.',
  tier: 1,
  difficulty: 3,
  estimatedHours: 3,
  status: 'planned',
  prerequisites: [
    {
      id: 'kl-jensen',
      strength: 'required',
      anchor: 'gibbs-inequality',
    },
    {
      id: 'em',
      strength: 'recommended',
      anchor: 'q-function',
    },
  ],
  recommendedNext: ['vae', 'gaussian-cookbook'],
  alsoUsedBy: ['vae', 'ddpm'],
  description:
    'The Evidence Lower Bound (ELBO) is the central objective of modern ' +
    'probabilistic ML. This lesson derives it two ways, shows the ' +
    'fundamental identity log p(x) = ELBO + KL, and revisits EM as ' +
    'coordinate ascent on this surface.',
  exportedAnchors: {
    'fundamental-identity':  'The fundamental identity: log p(x) = ELBO + KL',
    'elbo-two-forms':        'Two equivalent forms of the ELBO',
    'reparam-trick':         'The reparameterization trick (sketch)',
    'em-as-elbo-ascent':     'EM is coordinate ascent on the ELBO',
    'bimodal-failure':       'VI fails on bimodal posteriors with unimodal q',
  },
  path: '/lessons/elbo-vi',
};
```

---

## 4. Section-by-Section Plan

Eight sections. Plan for ~75–90 minutes of careful reading.

---

### Section 1 — Hook

**Length**: ~150 words. Full-width banner; SVG of a "ceiling" ($\log p(x)$, in
`--evidence`) with a rising "floor" (ELBO, in `--elbo`) approaching
it from below, and the gap (in `--kl-gap`) shrinking.

**Prose** (verbatim):

> The integral $p(x) = \int p(x, z) \, dz$ that defines the marginal
> likelihood — the *evidence* — is the central quantity in
> probabilistic modeling. It's also, for nearly every model worth
> caring about, **intractable**.
>
> Variational inference replaces this intractable computation with an
> optimization problem. We pick a tractable family of distributions
> $q$, find one that's "close" to the true posterior, and use it as
> a stand-in. The objective we maximize is called the **Evidence
> Lower Bound** — ELBO for short — and it's a lower bound on
> $\log p(x)$ that's tight when $q$ matches the posterior exactly.
>
> By the end of this lesson you'll have the fundamental identity in
> your back pocket, you'll know what VI does and what it can't do,
> and you'll see something nice: **the EM algorithm is exactly
> coordinate ascent on the ELBO**.

CTA button: "Set up the problem →"

---

### Section 2 — The Setup

**Length**: ~450 words. Establishes the latent-variable picture
formally before any ELBO machinery.

**Prose**:

> #### Latent-variable models
>
> A **latent-variable model** has two kinds of random variables:
>
> - $X$ — **observed**. We have data points $x_1, x_2, \ldots, x_N$.
> - $Z$ — **latent** (hidden). Never observed.
>
> The model specifies the **joint distribution** $p(x, z)$, typically
> factored as a prior on the latent and a conditional on the
> observation:
>
> $$p(x, z) \;=\; p(z) \, p(x \mid z)$$
>
> Two distributions derived from the joint matter for everything we'll
> do:
>
> - **Marginal likelihood** (a.k.a. **evidence**):
>   $$p(x) \;=\; \int p(x, z) \, dz$$
>   The probability the model assigns to the observation $x$, summing
>   over all possible latents.
>
> - **Posterior** over the latent given the observation:
>   $$p(z \mid x) \;=\; \frac{p(x, z)}{p(x)}$$
>   What we believe about $Z$ after seeing $x$.
>
> #### The intractability
>
> For all but the simplest models, **both** $p(x)$ and $p(z \mid x)$
> are intractable. They're the same computation: $p(x)$ is the
> integral; $p(z \mid x)$ is the integrand divided by the integral.
> Either way, an unevaluated $\int p(x, z) \, dz$ is sitting in the
> denominator.
>
> The intractability is structural, not a matter of cleverness:
>
> - For $z \in \mathbb{R}^d$ with $d$ large, the integral is
>   high-dimensional and lacks closed form except in special
>   conjugate cases.
> - For $z$ discrete with $K$ states per dimension and $D$
>   dimensions, the sum has $K^D$ terms.

> #### What variational inference does
>
> VI replaces these intractable computations with optimization:
>
> 1. Pick a **variational family** $\mathcal{Q}$ of tractable
>    distributions $q(z)$ — say, Gaussians.
> 2. Find $q^* \in \mathcal{Q}$ that is "close" to $p(z \mid x)$ in some
>    sense.
> 3. Use $q^*$ as a stand-in for the posterior wherever needed.
>
> The two design choices are: **what family $\mathcal{Q}$ to pick**,
> and **how to measure "close"**. The next section answers the second
> question.

**Cross-link callout — back to KL & Jensen** (`type=back`):

> **Uses: [KL & Jensen §4](../kl-jensen/#kl-definition)** for the KL
> divergence definition, and **[§7](../kl-jensen/#reverse-kl)** for the
> forward-vs-reverse intuition that drives the choice we'll make below.

---

### Section 3 — The ELBO Identity

**Length**: ~700 words. **Anchor: `fundamental-identity`**. The most
important section in the lesson.

**Prose**:

> We'll derive the ELBO two ways. Both are short. The second is more
> informative; the first is the historical motivation for the name.

> #### Derivation 1 — via Jensen's inequality
>
> Start from the marginal likelihood. Insert any distribution $q(z)$
> on the same support as $p(z \mid x)$ — multiply and divide, since
> the choice is free:
>
> $$\log p(x) \;=\; \log \int p(x, z) \, dz \;=\; \log \int q(z) \, \frac{p(x, z)}{q(z)} \, dz \;=\; \log \mathbb{E}_{Z \sim q}\!\left[\frac{p(x, Z)}{q(Z)}\right]$$
>
> Apply Jensen's inequality. Since $\log$ is concave, Jensen reverses:
>
> $$\log \mathbb{E}_{Z \sim q}\!\left[\frac{p(x, Z)}{q(Z)}\right] \;\geq\; \mathbb{E}_{Z \sim q}\!\left[\log \frac{p(x, Z)}{q(Z)}\right]$$
>
> Define the right-hand side to be the **Evidence Lower Bound**:
>
> $$\boxed{\;\; \mathrm{ELBO}(q) \;:=\; \mathbb{E}_{Z \sim q}\!\left[\log \frac{p(x, Z)}{q(Z)}\right] \;=\; \mathbb{E}_q[\log p(x, Z)] - \mathbb{E}_q[\log q(Z)] \;\;}$$
>
> What we just established is that for *any* $q$,
>
> $$\log p(x) \;\geq\; \mathrm{ELBO}(q)$$
>
> ELBO is a lower bound on the log evidence. Hence the name.

> #### Derivation 2 — via the KL identity
>
> Start from the KL divergence between $q$ and the true posterior:
>
> $$D_{\mathrm{KL}}(q(z) \,\|\, p(z \mid x)) \;=\; \mathbb{E}_q\!\left[\log \frac{q(Z)}{p(Z \mid x)}\right] \;=\; \mathbb{E}_q[\log q(Z)] - \mathbb{E}_q[\log p(Z \mid x)]$$
>
> Substitute $p(z \mid x) = p(x, z) / p(x)$:
>
> $$\mathbb{E}_q[\log p(Z \mid x)] \;=\; \mathbb{E}_q[\log p(x, Z)] - \log p(x)$$
>
> ($\log p(x)$ is constant in $Z$ so it pulls outside the expectation.)
>
> Plugging in:
>
> $$D_{\mathrm{KL}}(q \,\|\, p_{\text{post}}) \;=\; \mathbb{E}_q[\log q(Z)] - \mathbb{E}_q[\log p(x, Z)] + \log p(x) \;=\; -\mathrm{ELBO}(q) + \log p(x)$$
>
> Rearranging gives **the fundamental identity**:
>
> $$\boxed{\;\; \log p(x) \;=\; \mathrm{ELBO}(q) \;+\; D_{\mathrm{KL}}(q(z) \,\|\, p(z \mid x)) \;\;}$$
>
> Stop and read this twice. It says:
>
> 1. $\mathrm{ELBO}(q) \leq \log p(x)$ for every $q$ (since KL ≥ 0).
> 2. **The gap is exactly $D_{\mathrm{KL}}(q \,\|\, p_{\text{posterior}})$.**
> 3. Maximizing ELBO over $q$ is equivalent to minimizing reverse KL
>    to the posterior — and at the maximum, ELBO = $\log p(x)$ if
>    and only if $q = p_{\text{posterior}}$ (almost everywhere).

**Callout — "Why this identity is so useful"** (`type=tip`):

> Most of the time, we cannot compute $\log p(x)$ (that was the whole
> problem). But we can compute $\mathrm{ELBO}(q)$ — it's an
> expectation under our chosen $q$, of quantities we can evaluate
> ($\log p(x, z)$ is the joint, which we know, and $\log q(z)$ is
> trivial since we picked $q$). So the identity gives us a
> **handle**: maximize an evaluable quantity (ELBO) and you've
> implicitly done two things — you've **found** a good $q$ and you've
> **lower-bounded** $\log p(x)$ by a known quantity.

**Cross-link callout — back to KL & Jensen** (`type=back`):

> **Uses: [Jensen's inequality](../kl-jensen/#jensen-statement)** for
> Derivation 1. **Uses: [Gibbs' inequality](../kl-jensen/#gibbs-inequality)**
> implicitly in Derivation 2 (the gap $D_{\mathrm{KL}} \geq 0$ gives
> ELBO $\leq \log p(x)$). The fundamental identity is the structural
> reason variational inference works at all.

**Visualization 1 — `<ELBOKLDecomposition>`** (full width). The
canonical ELBO picture:

The visualization shows three quantities as a function of a 1D
parameter $\phi$ that controls $q$ (e.g., the mean of $q$ when its
variance is fixed):

- A horizontal line at $\log p(x)$ in `--evidence`.
- A curve $\mathrm{ELBO}(\phi)$ in `--elbo`, always below.
- A curve $D_{\mathrm{KL}}(q_\phi \,\|\, p_{\text{post}})$ in
  `--kl-gap`, always above zero. This is exactly $\log p(x) - \mathrm{ELBO}(\phi)$.

The example used: a 1D Gaussian posterior $\mathcal{N}(1, 1)$ (drawn
in a separate inset) and a variational $q_\phi = \mathcal{N}(\phi, 1)$.

A draggable point on the x-axis (current $\phi$ value) shows three
markers:
- Where ELBO sits at this $\phi$
- Where KL sits at this $\phi$
- A vertical bar visually decomposing $\log p(x)$ into ELBO + KL

As the user drags $\phi$ toward 1 (the posterior mean), ELBO climbs
toward $\log p(x)$, KL shrinks toward 0. **The reader sees the
identity's two terms trading off in real time.**

A button "Optimize" runs gradient ascent on ELBO; the marker animates
to $\phi = 1$.

---

### Section 4 — Two Forms of the ELBO

**Length**: ~400 words. **Anchor: `elbo-two-forms`**.

**Prose**:

> The ELBO has two forms that look different but are equal. Knowing
> both is essential — different parts of the literature use different
> ones, and each makes a different intuition obvious.

> #### Form 1 — joint form (what we derived)
>
> $$\mathrm{ELBO}(q) \;=\; \mathbb{E}_q[\log p(x, Z)] \;-\; \mathbb{E}_q[\log q(Z)]$$
>
> Define the entropy of $q$ as $H(q) := -\mathbb{E}_q[\log q(Z)]$:
>
> $$\mathrm{ELBO}(q) \;=\; \mathbb{E}_q[\log p(x, Z)] \;+\; H(q)$$
>
> **Interpretation**: pick $q$ to put mass on $z$ values where the
> joint $p(x, z)$ is large, but keep $q$'s entropy high (don't
> collapse to a delta function). Trade-off between fitting the joint
> and staying spread out.

> #### Form 2 — reconstruction-KL form (the VAE form)
>
> Use $\log p(x, z) = \log p(x \mid z) + \log p(z)$:
>
> $$\mathbb{E}_q[\log p(x, Z)] - \mathbb{E}_q[\log q(Z)] \;=\; \mathbb{E}_q[\log p(x \mid Z)] + \mathbb{E}_q[\log p(Z)] - \mathbb{E}_q[\log q(Z)]$$
>
> Group the last two terms: $\mathbb{E}_q[\log p(Z)] - \mathbb{E}_q[\log q(Z)] = -\mathbb{E}_q[\log(q(Z)/p(Z))] = -D_{\mathrm{KL}}(q(z) \,\|\, p(z))$.
>
> Therefore:
>
> $$\boxed{\;\; \mathrm{ELBO}(q) \;=\; \underbrace{\mathbb{E}_q[\log p(x \mid Z)]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}(q(z) \,\|\, p(z))}_{\text{regularizer}} \;\;}$$
>
> **Interpretation**: pick $q$ to make the observation $x$ likely
> when $z \sim q$ (good *reconstruction*), but keep $q$ close to the
> *prior* $p(z)$ (don't drift away from what the model thought $z$
> should look like).
>
> This is **exactly** the VAE training objective. The encoder
> $q_\phi(z \mid x)$ produces a distribution over latents, the decoder
> $p_\theta(x \mid z)$ scores how well the latent reconstructs $x$,
> and the KL term keeps the latents structured.

**Callout — "The two interpretations are both true"** (`type=tip`):

> Form 1 says: balance fitting the joint against entropy of $q$.
> Form 2 says: balance reconstruction against staying near the prior.
> They're algebraically the same — different ways of grouping the same
> three terms. Form 2 dominates the deep-learning literature because
> the prior $p(z)$ is usually a fixed simple thing (a standard
> Gaussian), so the KL is closed form.

**Cross-link callout — forward to VAE** (`type=forward`):

> **Comes back in: [Variational Autoencoders](../vae/) (planned)**
>
> The VAE training loss is, term-for-term, Form 2 of the ELBO with the
> sign flipped. The encoder amortizes the variational distribution
> over examples; the reparameterization trick (§5 below) makes the
> reconstruction term differentiable.

---

### Section 5 — Variational Inference Algorithm

**Length**: ~500 words. **Anchor: `reparam-trick`**.

**Prose**:

> So we want to maximize $\mathrm{ELBO}(q)$ over a tractable family
> $\mathcal{Q}$. There are three standard ways.
>
> #### 5a. Mean-field VI with closed-form updates (CAVI)
>
> Assume $q$ factorizes: $q(z) = \prod_{i=1}^d q_i(z_i)$. This is the
> **mean-field assumption**. Then the ELBO becomes a sum of pieces,
> and a coordinate-ascent update for each $q_i$ has a closed form:
>
> $$q_j^*(z_j) \;\propto\; \exp\Big(\mathbb{E}_{q_{-j}}\!\big[\log p(x, z)\big]\Big)$$
>
> where $q_{-j}$ means all the other factors. For models in the
> exponential family with conjugate priors, the update is simple
> arithmetic over natural parameters. (Derivation: take the
> functional derivative of ELBO with respect to $q_j$ subject to
> $\int q_j = 1$, set to zero. Sketch only — not the focus of this
> lesson.)
>
> CAVI is **exact in its updates** but the mean-field assumption is
> a strong restriction. It breaks correlations between latent
> dimensions.

> #### 5b. Gradient-based VI with parametric q
>
> Parametrize $q$ by $\phi$, e.g., $q_\phi(z) = \mathcal{N}(\mu_\phi, \Sigma_\phi)$.
> Compute $\nabla_\phi \mathrm{ELBO}$ and ascend.
>
> The reconstruction term $\mathbb{E}_{q_\phi}[\log p(x \mid Z)]$ is
> the awkward one — its gradient with respect to $\phi$ requires
> differentiating through a sample from $q_\phi$. The standard trick:
>
> #### 5c. The reparameterization trick
>
> Suppose we can write $Z = g_\phi(\epsilon)$ where $\epsilon$ is
> drawn from a fixed noise distribution $p(\epsilon)$ that doesn't
> depend on $\phi$. (Example: for $q_\phi = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$,
> use $g_\phi(\epsilon) = \mu_\phi + \sigma_\phi \epsilon$ with
> $\epsilon \sim \mathcal{N}(0, 1)$.)
>
> Then for any function $f$:
>
> $$\mathbb{E}_{Z \sim q_\phi}[f(Z)] \;=\; \mathbb{E}_{\epsilon \sim p(\epsilon)}\!\left[f\big(g_\phi(\epsilon)\big)\right]$$
>
> The right-hand side has $\phi$ inside the expectand only — so the
> gradient passes inside:
>
> $$\nabla_\phi \mathbb{E}_{Z \sim q_\phi}[f(Z)] \;=\; \mathbb{E}_{\epsilon \sim p(\epsilon)}\!\left[\nabla_\phi f\big(g_\phi(\epsilon)\big)\right]$$
>
> A Monte Carlo estimate is one sample of $\epsilon$, plug into
> $\nabla_\phi f(g_\phi(\epsilon))$, done. **This is the trick that
> makes VAEs trainable end-to-end with backpropagation.**

> #### 5d. Black-box VI (REINFORCE / score-function estimators)
>
> When $g_\phi$ doesn't exist (e.g., $z$ is discrete), use the score-
> function gradient
> $\nabla_\phi \mathbb{E}_{q_\phi}[f(Z)] = \mathbb{E}_{q_\phi}[f(Z) \nabla_\phi \log q_\phi(Z)]$.
> Higher variance, but unbiased and applicable everywhere.

**Callout — "What we'll demonstrate"** (`type=tip`):

> The §6 worked examples use **CAVI** for the conjugate Gaussian
> case (closed-form updates) and **gradient ascent** with the
> reparameterization trick for the bimodal case. That's a small but
> representative slice of the VI toolkit.

---

### Section 6 — Worked Examples

**Length**: ~900 words across two acts.

#### Act 1 — Conjugate Gaussian (when VI is exact)

**Setup**:

> Take the simplest non-trivial model:
>
> - Prior: $z \sim \mathcal{N}(0, \tau^2)$
> - Likelihood: $x_i \mid z \sim \mathcal{N}(z, \sigma^2)$, $i = 1, \ldots, n$, conditionally i.i.d.
> - Observe: $x_1, \ldots, x_n$
>
> Use $\tau^2 = 1$, $\sigma^2 = 1$, $n = 3$, data $x = (2.5, 1.7, 3.1)$.
> So $\bar x = 2.4333$.
>
> #### True posterior (closed form)
>
> By conjugacy:
>
> $$z \mid x \;\sim\; \mathcal{N}(\mu_n, \sigma_n^2), \quad \sigma_n^2 = \left(\frac{1}{\tau^2} + \frac{n}{\sigma^2}\right)^{\!-1} = 0.25, \quad \mu_n = \sigma_n^2 \cdot \frac{n \bar x}{\sigma^2} = 1.825$$
>
> So the posterior is $\mathcal{N}(1.825, 0.25)$. The marginal
> log-evidence (computed numerically by integration) is
> $\log p(x) \approx -6.1637$.
>
> #### Variational fit
>
> Take $\mathcal{Q} = \{\mathcal{N}(\phi_\mu, \phi_\sigma^2)\}$.
> Compute the ELBO using Form 2:
>
> $$\mathrm{ELBO}(\phi) \;=\; \sum_{i=1}^{n} \mathbb{E}_{q_\phi}[\log \mathcal{N}(x_i; Z, \sigma^2)] \;-\; D_{\mathrm{KL}}\!\big(\mathcal{N}(\phi_\mu, \phi_\sigma^2) \,\big\|\, \mathcal{N}(0, \tau^2)\big)$$
>
> The expectation under $Z \sim \mathcal{N}(\phi_\mu, \phi_\sigma^2)$:
>
> $$\mathbb{E}_{q_\phi}[(x_i - Z)^2] \;=\; (x_i - \phi_\mu)^2 + \phi_\sigma^2$$
>
> So:
>
> $$\mathbb{E}_{q_\phi}[\log \mathcal{N}(x_i; Z, \sigma^2)] \;=\; -\tfrac{1}{2}\log(2\pi\sigma^2) \;-\; \frac{(x_i - \phi_\mu)^2 + \phi_\sigma^2}{2\sigma^2}$$
>
> The KL between Gaussians (from
> [KL & Jensen §4](../kl-jensen/#kl-gaussians)):
>
> $$D_{\mathrm{KL}}\!\big(\mathcal{N}(\phi_\mu, \phi_\sigma^2) \,\big\|\, \mathcal{N}(0, \tau^2)\big) \;=\; \tfrac{1}{2}\log \frac{\tau^2}{\phi_\sigma^2} \;+\; \frac{\phi_\sigma^2 + \phi_\mu^2}{2 \tau^2} \;-\; \tfrac{1}{2}$$
>
> #### Solving the ELBO
>
> Take partial derivatives.
>
> $\partial \mathrm{ELBO} / \partial \phi_\mu$: only the $(\phi_\mu - x_i)^2$
> terms and the $\phi_\mu^2/(2\tau^2)$ term contribute.
>
> $$\frac{\partial \mathrm{ELBO}}{\partial \phi_\mu} \;=\; \sum_{i=1}^{n} \frac{x_i - \phi_\mu}{\sigma^2} \;-\; \frac{\phi_\mu}{\tau^2} \;=\; \frac{n(\bar x - \phi_\mu)}{\sigma^2} \;-\; \frac{\phi_\mu}{\tau^2}$$
>
> Setting to zero and solving:
>
> $$\phi_\mu^* \;=\; \frac{n \bar x / \sigma^2}{n/\sigma^2 + 1/\tau^2} \;=\; \mu_n \;=\; 1.825$$
>
> $\partial \mathrm{ELBO} / \partial \phi_\sigma^2$:
>
> $$\frac{\partial \mathrm{ELBO}}{\partial \phi_\sigma^2} \;=\; -\frac{n}{2\sigma^2} \;+\; \frac{1}{2 \phi_\sigma^2} \;-\; \frac{1}{2\tau^2}$$
>
> Setting to zero:
>
> $$\frac{1}{\phi_\sigma^{*2}} \;=\; \frac{n}{\sigma^2} + \frac{1}{\tau^2} \;\;\Longrightarrow\;\; \phi_\sigma^{*2} \;=\; \sigma_n^2 \;=\; 0.25$$
>
> **The variational optimum exactly matches the posterior**:
> $q^* = \mathcal{N}(1.825, 0.25) = p(z \mid x)$. The KL gap is zero.
> ELBO at $\phi^*$ equals $\log p(x) = -6.1637$ exactly.
>
> This is the message of Act 1: **when the variational family
> contains the true posterior, VI is exact**, and the ELBO bound
> becomes an equality.

| $\phi$ | $\mathrm{ELBO}(\phi)$ | KL gap |
|:------:|:---------------------:|:------:|
| $q^* = \mathcal{N}(1.825, 0.25)$ | $-6.1637$ | $0.0000$ |
| $q = $ prior $\mathcal{N}(0, 1)$ | $-13.6318$ | $7.4681$ |
| $q = \mathcal{N}(\bar x, \sigma^2/n) = \mathcal{N}(2.43, 0.33)$ (MLE-ish) | $-6.9267$ | $0.7630$ |

(Verify in `gaussian.test.ts`. These are pre-computed correct values.)

**Visualization 2 — `<ELBOOptimization>`**: Two-panel display.

- Left panel: density plot of the true posterior in `--posterior`,
  overlaid with the current $q_\phi$ in `--variational`. Sliders for
  $\phi_\mu, \phi_\sigma^2$.
- Right panel: live readouts of ELBO, KL gap, $\log p(x)$. A
  horizontal-bar decomposition showing the ELBO + KL = $\log p(x)$
  identity as it currently stands.
- Buttons: "Set to prior" / "Set to posterior" / "Run gradient ascent".
  The last animates $\phi \to \phi^*$ over ~30 frames. ELBO climbs,
  KL shrinks.

#### Act 2 — Bimodal posterior (when VI fails gracefully)

**Anchor: `bimodal-failure`**.

**Setup**:

> Imagine a model whose posterior comes out to
>
> $$p(z \mid x) \;=\; \tfrac{1}{2}\mathcal{N}(z; -3, 1) \;+\; \tfrac{1}{2}\mathcal{N}(z; 3, 1)$$
>
> — bimodal, symmetric. (We constructed this so $\log p(x) = 0$
> exactly: the unnormalized joint is the sum of two normalized
> Gaussians, integrating to 1.)
>
> The variational family $\mathcal{Q} = \{\mathcal{N}(\phi_\mu, \phi_\sigma^2)\}$
> contains only **unimodal** Gaussians. The posterior is bimodal.
> The family does not contain the truth.

**Recognize this distribution.** It's the same bimodal target from
[KL & Jensen §7](../kl-jensen/#reverse-kl). Back there we computed
the reverse-KL fit numerically and got
$q^{\text{rev}} \approx \mathcal{N}(\pm 3, 1.05)$. **That's exactly the
ELBO maximum here.** Maximizing ELBO is minimizing reverse KL — same
optimization, two names.

**Worked numbers**:

| $q$ | $\mathrm{ELBO}(q)$ | KL gap |
|:---:|:------------------:|:------:|
| $q^* = \mathcal{N}(\pm 3, 1.05)$ (mode-seeking) | $-0.6888$ | $0.6888$ |
| $q = \mathcal{N}(0, 10)$ (forward-KL fit, mass-covering) | $-0.7789$ | $0.7789$ |
| $q = \mathcal{N}(0, 1)$ (centered, narrow) | $-2.6934$ | $2.6934$ |
| $q = $ true posterior $0.5\mathcal{N}(-3,1) + 0.5\mathcal{N}(3,1)$ (not in family) | $0.0000$ | $0.0000$ |

(Note the KL gap at the optimum is $\approx \log 2 = 0.693$ — close
to the entropy cost of "committing to one mode out of two equal
modes." A revealing back-of-envelope.)

**Cross-link callout — back to KL & Jensen** (`type=back`):

> **The same fit, two derivations.** In KL §7 we found the reverse-KL
> fit by direct minimization. Here we found it by maximizing ELBO.
> By the fundamental identity of §3, those are the same optimization
> — and indeed both give $q^* \approx \mathcal{N}(\pm 3, 1.05)$. VI is
> reverse-KL minimization in formal disguise.

**Visualization 3 — `<BimodalELBO>`**: The centerpiece. Layout:

```
┌────────────────────────────────────────────────────────────────────┐
│  TARGET: p(z|x) = 0.5·N(-3,1) + 0.5·N(3,1)                          │
│  [posterior shown in --posterior, fixed]                            │
└────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────┬───────────────────────────────────┐
│  VARIATIONAL FIT                 │  ELBO LANDSCAPE                    │
│                                  │                                   │
│  Sliders: phi_mu, phi_sigma^2    │  Heatmap of ELBO over             │
│                                  │  (phi_mu, phi_sigma^2) grid       │
│  [posterior + current q          │  with current point + trajectory  │
│   overlaid]                      │  of past optimization steps       │
│                                  │                                   │
│  Buttons: Reset, Init at -5,     │  log p(x) = 0.000                 │
│  Init at 0, Init at +5,          │  ELBO    = -0.6888 (current)      │
│  Run grad ascent                 │  KL gap  =  0.6888 (current)      │
└─────────────────────────────────┴───────────────────────────────────┘
```

The right panel (the ELBO landscape) is a 2D heatmap over $(\phi_\mu,
\log \phi_\sigma)$ with two visible maxima at $\phi_\mu = \pm 3$ and a
saddle along $\phi_\mu = 0$. Reaching either maximum gives the same
ELBO value by symmetry — a perfect demonstration of the local-optima
issue VI inherits from EM.

The "Init at 0" button starts at the saddle and shows that gradient
ascent (with no noise) stays at the saddle — same pathology as the
"Symmetric" preset in the EM simulator. **Pedagogical alignment with
EM is intentional.**

---

### Section 7 — EM Through the ELBO Lens

**Length**: ~700 words. **Anchor: `em-as-elbo-ascent`**. The big
re-framing.

**Prose**:

> Now we'll show that the EM algorithm — derived two lessons ago for
> the two-coins problem — is **literally coordinate ascent on the
> ELBO**. This re-framing replaces the EM proof with one line and
> generalizes EM to settings where the E-step can't be done exactly
> (which is, well, most of modern ML).
>
> #### Setup
>
> Now we have **two** sets of unknowns: the variational distribution
> $q$ and the model parameters $\theta$. The joint depends on $\theta$:
> $p_\theta(x, z) = p_\theta(z) p_\theta(x \mid z)$. The ELBO depends
> on both:
>
> $$\mathcal{L}(q, \theta) \;:=\; \mathbb{E}_q[\log p_\theta(x, Z)] - \mathbb{E}_q[\log q(Z)]$$
>
> The fundamental identity becomes:
>
> $$\log p_\theta(x) \;=\; \mathcal{L}(q, \theta) \;+\; D_{\mathrm{KL}}(q(z) \,\|\, p_\theta(z \mid x))$$
>
> #### Coordinate ascent
>
> Maximize $\mathcal{L}$ alternately, one variable at a time:
>
> **Step E** (optimize over $q$, holding $\theta$ fixed):
>
> $$q^{(t+1)} \;=\; \arg\max_q \; \mathcal{L}(q, \theta^{(t)})$$
>
> If we allow $q$ to be *any* distribution (no restriction to a
> tractable family), the maximum is at $q = p_{\theta^{(t)}}(z \mid x)$
> — the **exact posterior** — because that's where the KL gap closes.
> At the optimum,
>
> $$\mathcal{L}(q^{(t+1)}, \theta^{(t)}) \;=\; \log p_{\theta^{(t)}}(x)$$
>
> **Step M** (optimize over $\theta$, holding $q$ fixed):
>
> $$\theta^{(t+1)} \;=\; \arg\max_\theta \; \mathcal{L}(q^{(t+1)}, \theta) \;=\; \arg\max_\theta \; \mathbb{E}_{q^{(t+1)}}[\log p_\theta(x, Z)]$$
>
> (The entropy $-\mathbb{E}_{q}[\log q(Z)]$ doesn't depend on $\theta$,
> so it drops.)

> #### These are exactly the EM steps
>
> Compare to [EM §5](../em/#e-step):
>
> - The EM E-step computes responsibilities
>   $\gamma_i^A = P(Z_i = A \mid x_i, \theta^{(t)})$. **That's
>   $q^{(t+1)}(z) = p_{\theta^{(t)}}(z \mid x)$ for the two-coins
>   discrete case.** The "responsibilities" *are* the exact posterior.
> - The EM M-step maximizes
>   $Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{q^{(t+1)}}[\log p_\theta(x, Z)]$.
>   **That's the ELBO M-step.** The Q function is the ELBO with the
>   entropy term dropped (legal because it's constant in $\theta$).
>
> So EM is coordinate ascent on $\mathcal{L}(q, \theta)$.

> #### And the monotonicity proof becomes one line
>
> Coordinate ascent on a function never decreases its value. So
> $\mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \theta^{(t)})$.
> But after each E-step the bound is tight:
> $\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p_{\theta^{(t)}}(x)$.
> Combined:
>
> $$\log p_{\theta^{(t+1)}}(x) \;\geq\; \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \;\geq\; \mathcal{L}(q^{(t+1)}, \theta^{(t)}) \;=\; \log p_{\theta^{(t)}}(x)$$
>
> The middle step is the M-step's monotonicity ($\theta^{(t+1)}$
> maximizes $\mathcal{L}(q^{(t+1)}, \cdot)$). The first step is the
> fundamental identity ($\log p \geq \mathcal{L}$ always). The last
> equality is the post-E-step tightness. Three lines, no Gibbs'
> inequality needed at the level of $\theta$. **The EM lesson's
> Theorem 7.2.20 is now this one-paragraph argument.**

**Cross-link callout — back to EM** (`type=back`):

> **Re-derivation of: [EM §8 monotonicity proof](../em/#monotonicity)**
>
> The original EM proof factored
> $\ell(\theta^{(t+1)}) - \ell(\theta^{(t)})$ into a Q-gap (M-step) and a
> KL-gap (Gibbs' inequality on the conditional density). That proof is
> correct but elaborate. The ELBO view collapses it to "coordinate
> ascent on $\mathcal{L}$ with tight bound after each E-step."

**Worked numbers — re-running the 2-coins ELBO**:

> Let's verify on the two-coins example. Initialize
> $\theta^{(0)} = (0.6, 0.5)$ and $q^{(0)}$ = uninformative
> (each $\gamma_i = 0.5$).
>
> | step | what happened | $\mathcal{L}(q, \theta)$ | $\log p_\theta(x)$ | KL gap |
> |:----:|:-------------:|:------------------------:|:------------------:|:------:|
> | 0 | initialized | $-33.5458$ | $-33.0939$ | $0.4519$ |
> | 0+E | E-step at $\theta^{(0)}$ | $-33.0939$ | $-33.0939$ | $0.0000$ |
> | 0+M | M-step → $\theta^{(1)} \approx (0.713, 0.581)$ | $-31.9972$ | $-31.8593$ | $0.1380$ |
> | 1+E | E-step at $\theta^{(1)}$ | $-31.8593$ | $-31.8593$ | $0.0000$ |
>
> Read the table: at every E-step the ELBO **snaps up to meet
> $\log p(x)$**. At every M-step both rise, but the ELBO trails
> $\log p(x)$ slightly because $q$ is no longer the exact posterior
> for the new $\theta$. The dance continues until convergence.

**Visualization 4 — `<EMTrajectory>`** (medium width):

A line chart with iteration on the x-axis (with sub-iterations marked
as "0", "0+E", "0+M", "1+E", …) and log-likelihood on the y-axis.
Two lines:
- $\log p_{\theta^{(t)}}(x)$ in `--evidence` — flat during E-steps,
  jumps up during M-steps.
- $\mathcal{L}(q, \theta)$ in `--elbo` — jumps up during E-steps to
  meet the evidence line, then both move together (with small gaps)
  during M-steps.

The reader sees the "saw-tooth" pattern of the bound being tightened
and re-opened. **This is the visual proof of EM monotonicity.**

**Callout — "When you can't do the E-step exactly"** (`type=tip`):

> If the variational family $\mathcal{Q}$ doesn't contain the exact
> posterior, the E-step is "best $q$ in $\mathcal{Q}$" — the bound
> doesn't fully tighten. This is **variational EM** (or, if
> generalized further, just plain VI). Modern deep generative models
> (VAE, DDPM) are in this regime. The KL gap is the price of
> tractability.

---

### Section 8 — Where You'll See This

**Length**: ~250 words.

**Prose**:

> The ELBO and its identity are everywhere in modern probabilistic
> ML. A short list of where, with links into the lessons that build
> on this one.
>
> #### Already used
>
> - **EM** ([revisited above](../em/#monotonicity)). EM is exact
>   variational EM where the E-step achieves the bound's tightness.
>
> - **KL & Jensen** (the foundations). The fundamental identity is
>   Jensen-on-log; the gap is reverse KL; both come from there.
>
> #### Coming next
>
> - **Variational Autoencoders.** Form 2 of the ELBO is the VAE loss.
>   The encoder amortizes $q_\phi(z \mid x)$ across data points. The
>   reparameterization trick (§5c) makes the reconstruction
>   differentiable. The KL term keeps latents structured.
>
> - **DDPM.** The DDPM training objective is a sum of per-timestep
>   ELBOs. Each timestep contributes a KL between two Gaussians
>   (using the [closed form](../kl-jensen/#kl-gaussians) we derived).
>   The chain rule of KL decomposes the joint ELBO across diffusion
>   steps.
>
> #### Adjacent ideas (sidebar)
>
> - **Mean-field VI** for graphical models (CAVI, exact updates).
>   Bishop ch. 10 has the canonical treatment.
> - **Black-box VI** for non-conjugate models (Ranganath, Gerrish,
>   Blei 2014).
> - **β-VAE** and information-bottleneck variants — controlling the
>   KL coefficient changes representation properties.
> - **Importance-weighted ELBO** (IWAE) — tighter bounds via multiple
>   $q$-samples.

**Visualization 7 — `<RoadmapMini>`** highlighting the current lesson
and showing that VAE and DDPM both depend on it.

---

## 5. Algorithm / Math Implementation

### `src/lessons/elbo-vi/math/gaussian.ts` — Act 1 (closed form)

```ts
import { klGaussian } from '@lessons/kl-jensen/math/kl';

export interface GaussianConjugate {
  tau2: number;
  sigma2: number;
  data: number[];
}

export function posterior(model: GaussianConjugate): { mu: number; var_: number } {
  const n = model.data.length;
  const xbar = model.data.reduce((a, b) => a + b, 0) / n;
  const var_ = 1 / (1 / model.tau2 + n / model.sigma2);
  const mu = var_ * (n * xbar / model.sigma2);
  return { mu, var_ };
}

/** Compute log p(x) by numerical integration (truth value for tests). */
export function logEvidence(model: GaussianConjugate, gridSize = 10000): number {
  const zs: number[] = [];
  const a = -10, b = 10;
  for (let i = 0; i < gridSize; i++) zs.push(a + (b - a) * i / (gridSize - 1));
  const dz = zs[1] - zs[0];
  const logJoint = zs.map(z => {
    let logPrior = -0.5 * Math.log(2 * Math.PI * model.tau2) - z*z / (2 * model.tau2);
    let logLik = 0;
    for (const x of model.data) {
      logLik += -0.5 * Math.log(2 * Math.PI * model.sigma2)
              - (x - z)*(x - z) / (2 * model.sigma2);
    }
    return logPrior + logLik;
  });
  const m = Math.max(...logJoint);
  let s = 0;
  for (const lj of logJoint) s += Math.exp(lj - m);
  return m + Math.log(s * dz);
}

/** ELBO at q = N(phi_mu, phi_var) under Form 2. */
export function elboGaussian(model: GaussianConjugate, phi_mu: number, phi_var: number): number {
  const n = model.data.length;
  let recon = 0;
  for (const x of model.data) {
    recon += -0.5 * Math.log(2 * Math.PI * model.sigma2)
           - ((x - phi_mu)*(x - phi_mu) + phi_var) / (2 * model.sigma2);
  }
  const kl = klGaussian(phi_mu, Math.sqrt(phi_var), 0, Math.sqrt(model.tau2));
  return recon - kl;
}
```

### `src/lessons/elbo-vi/math/bimodal.ts` — Act 2 (numerical)

```ts
/** Unnormalized posterior: 0.5 N(z; -3, 1) + 0.5 N(z; 3, 1). */
export function bimodalPosterior(z: number): number {
  const g1 = Math.exp(-(z + 3) ** 2 / 2) / Math.sqrt(2 * Math.PI);
  const g2 = Math.exp(-(z - 3) ** 2 / 2) / Math.sqrt(2 * Math.PI);
  return 0.5 * g1 + 0.5 * g2;
}

/** ELBO at q = N(phi_mu, phi_var) for the bimodal posterior, by numerical integration. */
export function elboBimodal(phi_mu: number, phi_var: number, gridSize = 4000): number {
  const a = -8, b = 8;
  const zs: number[] = [];
  for (let i = 0; i < gridSize; i++) zs.push(a + (b - a) * i / (gridSize - 1));
  const dz = zs[1] - zs[0];
  let s = 0;
  for (const z of zs) {
    const pUnnorm = bimodalPosterior(z);
    const q = Math.exp(-(z - phi_mu) ** 2 / (2 * phi_var)) / Math.sqrt(2 * Math.PI * phi_var);
    if (q < 1e-30) continue;
    const logRatio = Math.log(Math.max(pUnnorm, 1e-30)) - Math.log(q);
    s += q * logRatio * dz;
  }
  return s;
}
```

### `src/lessons/elbo-vi/math/em-elbo.ts` — Act 3 (2-coins)

```ts
import { TRIALS, FLIPS_PER_TRIAL } from '@lessons/em/em/data';

const M = FLIPS_PER_TRIAL;

/** ELBO L(q, theta) for the 2-coins model with q factorized as Bernoulli(gamma_i). */
export function elboTwoCoins(gammas: number[], thetaA: number, thetaB: number): number {
  let s = -TRIALS.length * Math.log(2);  // sum of log(1/2) for the priors
  for (let i = 0; i < TRIALS.length; i++) {
    const { heads: h, tails: t } = TRIALS[i];
    const g = gammas[i];
    s += g * (h * Math.log(thetaA) + t * Math.log(1 - thetaA))
       + (1 - g) * (h * Math.log(thetaB) + t * Math.log(1 - thetaB));
    if (g > 0 && g < 1) {
      s += -g * Math.log(g) - (1 - g) * Math.log(1 - g);  // entropy
    }
  }
  return s;
}
```

### Test cases

- `posterior({tau2: 1, sigma2: 1, data: [2.5, 1.7, 3.1]})` →
  `{ mu: 1.825, var_: 0.25 }`
- `logEvidence({tau2: 1, sigma2: 1, data: [2.5, 1.7, 3.1]})` ≈ −6.1637
- `elboGaussian(model, 1.825, 0.25)` ≈ −6.1637 (matches log evidence)
- `elboGaussian(model, 0, 1)` ≈ −13.6318
- `elboGaussian(model, 2.4333, 0.3333)` ≈ −6.9267
- `elboBimodal(3, 1.05)` ≈ −0.6888
- `elboBimodal(0, 10)` ≈ −0.7789
- `elboBimodal(0, 1)` ≈ −2.6934
- `elboTwoCoins([0.5, 0.5, 0.5, 0.5, 0.5], 0.6, 0.5)` ≈ −33.5458
- `elboTwoCoins([0.4491, 0.8050, 0.7335, 0.3522, 0.6472], 0.6, 0.5)` ≈ −33.0939
  (matches `observedLogLikelihood(0.6, 0.5)` from the EM lesson — bound
  is tight after E-step)
- `elboTwoCoins([0.4491, 0.8050, 0.7335, 0.3522, 0.6472], 0.7130, 0.5813)` ≈ −31.9972

---

## 6. Component Catalog

### Shared (already exist in `@shared/ui/`)
- `<NavigationSidebar>`, `<ProgressBar>`, `<PrereqStrip>`,
  `<CrosslinkCallout>`, `<RoadmapMini>`, `<MathInline>`, `<MathBlock>`,
  `<Callout>`, `<DataTable>`, `<ProofToggle>`.

### Visualizations (lesson-local in `viz/`)
- `<ELBOKLDecomposition>` (§3) — the fundamental-identity centerpiece.
- `<ELBOOptimization>` (§6 Act 1) — Gaussian conjugate.
- `<BimodalELBO>` (§6 Act 2) — bimodal failure mode + ELBO landscape.
- `<EMTrajectory>` (§7) — sawtooth ELBO during EM iterations.
- `<PosteriorComparison>` — small reusable component for the
  posterior/q overlay used in Acts 1 and 2.

---

## 7. Page-Level UX

Same as the other lessons. `<PrereqStrip>` at the top showing required
prereq KL & Jensen and recommended prereq EM. Sticky sidebar with §1–§8.
Top progress bar. Keyboard navigation. Reduced-motion respected on the
optimization animations in §6 and §7. `<RoadmapMini>` at the bottom.

The §6 Act 2 visualization (the centerpiece) and the §7 EM trajectory
should be especially polished — these are the "wow" moments.

---

## 8. Acceptance Criteria

A learner who has worked through this page should be able to, on a
blank sheet:

1. Write the fundamental identity
   $\log p(x) = \mathrm{ELBO}(q) + D_{\mathrm{KL}}(q \,\|\, p_{\text{post}})$
   from memory, and explain what each term represents.
2. Derive the ELBO via Jensen's inequality starting from
   $\log p(x) = \log \int q(z) (p(x,z)/q(z)) dz$.
3. Convert between the joint form and the reconstruction-KL form of
   ELBO — derivation shown.
4. Explain what the reparameterization trick is and why it makes
   gradient-based VI tractable for Gaussian $q$.
5. Explain why VI is **reverse-KL** minimization, and predict
   (qualitatively) the failure mode on multimodal posteriors.
6. Show that EM's E-step and M-step are coordinate ascent on
   $\mathcal{L}(q, \theta)$, with the E-step achieving tightness.
7. Compute $\mathrm{ELBO}$ for the conjugate Gaussian example by
   hand or with a calculator and verify it matches $\log p(x)$ at
   the optimum.

If a friendly TA quizzed the learner on the above and they failed two
or more, the page didn't do its job.

---

## 9. Stretch Goals (post-MVP)

- **Mean-field VI walkthrough**: a small CAVI demo on a 2D Gaussian
  posterior — diagonal mean-field $q$ recovers correct marginals but
  underestimates correlation.
- **Importance-weighted ELBO (IWAE)**: a slider for $K$ (number of
  importance samples) showing how the bound tightens as $K$ grows.
- **β-VAE**: scale the KL term by $\beta$; show how $\beta > 1$
  leads to disentangled-but-uninformative $q$ and $\beta < 1$ leads
  to high-fidelity-but-irregular latents. (Defer mostly to VAE lesson.)
- **Score-function gradient demo**: side-by-side variance comparison
  with reparameterization. Shows why reparam is preferred when
  available.

---

## 10. Out of Scope (intentionally)

- **Multivariate Gaussian KL** in full matrix form — defer to
  Gaussian Cookbook lesson, where it lives natively.
- **Convergence guarantees** for stochastic VI — deep, technical,
  not on the path to DDPM.
- **Amortized inference** beyond a sentence — that's the
  introduction to VAE.
- **The MCMC/VI tradeoff** — adjacent and interesting, not on the
  main thread.
- **Natural gradients / information geometry** — beautiful but tangential.