# EM Algorithm — Interactive Lesson
## Build Specification & Content Plan

> A single-page, deeply explained, interactive lesson on the Expectation-Maximization
> algorithm. Uses the **two-coins problem** as its single running example. Targets a
> learner who knows basic probability/MLE and wants to *actually understand* EM —
> not just be told what it does.

---

## 0. Pedagogical Philosophy

The page is structured so a learner can follow it linearly without ever feeling
"trust me on this." Three commitments:

1. **Concrete before abstract.** The two-coins setup is introduced before any of
   the EM machinery so the learner has something to anchor every formula to.
2. **Math is shown in full.** Every derivation step appears. No "after some
   algebra…" The page should pass the test of a smart reader who refuses to
   accept a step they didn't see.
3. **Visualization carries half the load.** Each abstract object — the
   responsibilities, the likelihood surface, the lower-bound trick — gets a
   dedicated visualization that the reader can manipulate.

The reader must, by the end, be able to (a) state the responsibility formula
from memory, (b) write out the M-step update, (c) sketch why the observed
log-likelihood never decreases, and (d) predict EM's behavior under different
initializations.

---

## 1. Tech Stack

- **Build**: Vite + TypeScript (no framework needed; vanilla TS modules + ES
  modules is enough). If the implementer prefers, Preact (10 kB) is fine — but
  React is overkill.
- **Math**: KaTeX (CDN-loadable, fast). Auto-render plugin handles `$…$` and
  `$$…$$` delimiters in the prose.
- **Plotting/visualization**: D3.js v7 for everything custom (likelihood
  contour, responsibility bars, trajectory overlays). No Chart.js — the
  bespoke visualizations don't fit a chart library template.
- **Styling**: Hand-written CSS with custom properties for tokens. No Tailwind.
- **Animation**: CSS transitions for hovers and toggles; D3 transitions for
  chart updates; the Web Animations API for the iteration trajectory replay.

Suggested file layout:

```
em-lesson/
├── index.html
├── src/
│   ├── main.ts                  # entry, builds page from sections
│   ├── katex-render.ts          # invokes KaTeX auto-render after DOM mount
│   ├── em/
│   │   ├── data.ts              # the 5 trials, hardcoded
│   │   ├── algorithm.ts         # eStep, mStep, observedLogLik, runEM (pure)
│   │   └── algorithm.test.ts    # sanity checks (optional but recommended)
│   ├── sections/
│   │   ├── 01-hook.ts
│   │   ├── 02-the-problem.ts
│   │   ├── 03-complete-vs-incomplete.ts
│   │   ├── 04-the-key-idea.ts
│   │   ├── 05-e-step.ts
│   │   ├── 06-m-step.ts
│   │   ├── 07-full-algorithm.ts
│   │   ├── 08-convergence.ts
│   │   ├── 09-pitfalls.ts
│   │   └── 10-summary.ts
│   ├── viz/
│   │   ├── binomial-mixture.ts
│   │   ├── likelihood-surface.ts
│   │   ├── responsibility-calculator.ts
│   │   ├── elbo-diagram.ts
│   │   ├── em-simulator.ts          # the centerpiece
│   │   ├── monotonicity-demo.ts
│   │   └── multi-restart-gallery.ts
│   ├── ui/
│   │   ├── callout.ts               # styled aside boxes
│   │   ├── proof-toggle.ts          # collapsible proof
│   │   ├── sidebar.ts               # sticky TOC
│   │   └── progress-bar.ts          # mobile reading progress
│   └── styles/
│       ├── tokens.css               # color/typography variables
│       ├── prose.css                # body text, headings
│       ├── components.css           # buttons, callouts, etc
│       └── viz.css                  # visualization-specific
└── package.json
```

Each section file exports a function `mount(container: HTMLElement): void`
that injects its HTML and wires up its viz components.

---

## 2. Visual / Aesthetic Direction

The page should feel like a beautifully typeset academic essay that happens to
have things you can poke. Reference points: distill.pub, Bartosz Ciechanowski's
explorables, Nicky Case. Calm, generous, paper-like — not a SaaS dashboard.

**Color tokens** (warm, paper-like; defined in `tokens.css`):

```css
:root {
  --paper:        #faf6ee;   /* page background, warm off-white */
  --paper-soft:   #f3edde;   /* callout backgrounds, table rows */
  --ink:          #1a1a1a;   /* primary text */
  --ink-soft:     #555148;   /* secondary text, captions */
  --rule:         #d4ccbf;   /* dividers, faint borders */
  --coin-a:       #c0392b;   /* Coin A — terracotta */
  --coin-b:       #2c5f8d;   /* Coin B — slate blue */
  --amber:        #d4a437;   /* equation highlight, key boxed eqs */
  --sage:         #5a8a6a;   /* convergence, success */
  --shadow:       rgba(40, 30, 15, 0.08);
}
@media (prefers-color-scheme: dark) {
  :root { --paper: #1a1814; --paper-soft: #232017; --ink: #ece6d6;
          --ink-soft: #a09680; --rule: #3a3528; --shadow: rgba(0,0,0,0.4); }
}
```

**Typography** (Google Fonts, all loaded from a single `<link>`):

- Display headings: `Fraunces` (variable serif, italic optical sizes)
- Body prose: `Source Serif 4` (clean serif at 18px, line-height 1.65)
- KaTeX defaults (Computer Modern) — pairs naturally with the body serif
- Code & numerics: `JetBrains Mono` at 0.92em

Body width: max 660px for prose blocks. Visualizations span up to 920px and
are centered. The first paragraph of each section gets a drop cap. Generous
whitespace between sections (≈4rem).

---

## 3. The Two-Coins Setup (used everywhere)

This is the one example. Hardcode it in `src/em/data.ts`:

```ts
export const TRIALS = [
  { id: 1, sequence: 'HTTTHHTHTH', heads: 5, tails: 5 },
  { id: 2, sequence: 'HHHHTHHHHH', heads: 9, tails: 1 },
  { id: 3, sequence: 'HTHHHHHTHH', heads: 8, tails: 2 },
  { id: 4, sequence: 'HTHTTTHHTT', heads: 4, tails: 6 },
  { id: 5, sequence: 'THHHTHHHTH', heads: 7, tails: 3 },
];
export const FLIPS_PER_TRIAL = 10;
export const PRIOR = 0.5; // P(coin = A) = 0.5
```

(This is the dataset from Do & Batzoglou, *Nature Biotechnology* 2008, "What is
the expectation maximization algorithm?" — a standard pedagogical choice. The
algorithm converges to roughly $\theta_A \approx 0.797$, $\theta_B \approx
0.520$ from initialization $(0.6, 0.5)$.)

---

## 4. Section-by-Section Plan

Each section spec below contains: **prose** (verbatim or near-verbatim),
**math** (in LaTeX, with all derivation steps), **visualizations**, and
**interactions**.

---

### Section 1 — Hook

**Length**: ~120 words, full-width banner with a small SVG illustration of two coins.

**Visual**: Two coin SVGs side by side, gently bobbing (CSS `@keyframes`,
respect `prefers-reduced-motion`). Coin A in `--coin-a`, Coin B in `--coin-b`.

**Prose** (verbatim):

> Imagine someone hands you a sequence of coin flip results — say, fifty
> heads-or-tails outcomes. You're told two coins were used, with different
> biases, and that for each block of ten flips one coin was chosen at random.
> But you're not told *which* coin was used for each block.
>
> Can you figure out how biased each coin is?
>
> This is a problem with **missing information** — and it's exactly the kind
> of problem the EM algorithm was built to solve. By the end of this lesson
> you'll know how it works, why it works, and you'll be able to watch it
> work, step by step.

CTA button at the bottom: "Let's set it up →" (smooth-scrolls to §2).

---

### Section 2 — The Problem

**Length**: ~450 words.

**Prose**:

> Let's pin the problem down. We have two coins, **Coin A** and **Coin B**,
> with unknown probabilities of landing heads $\theta_A$ and $\theta_B$. We
> run an experiment 5 times. Each time:
>
> 1. Pick one of the two coins uniformly at random (50/50). Call this choice
>    $Z_i \in \{A, B\}$.
> 2. Flip the chosen coin **10 times** and record the number of heads $x_i$.
> 3. Hide the value of $Z_i$ — we don't get to see which coin was used.
>
> So our observed data is $(x_1, x_2, \ldots, x_5)$. Our hidden ("missing")
> data is $(Z_1, Z_2, \ldots, Z_5)$. The parameters we want to estimate are
> $\theta = (\theta_A, \theta_B)$.

Then introduce the actual data as a styled table:

| Trial $i$ | Sequence | Heads $x_i$ | Tails $10 - x_i$ |
|:---------:|:---------|:-----------:|:----------------:|
| 1 | H T T T H H T H T H | 5 | 5 |
| 2 | H H H H T H H H H H | 9 | 1 |
| 3 | H T H H H H H T H H | 8 | 2 |
| 4 | H T H T T T H H T T | 4 | 6 |
| 5 | T H H H T H H H T H | 7 | 3 |

Continue:

> Squint at the table for a moment. Trial 2 (9 heads) and Trial 3 (8 heads)
> *feel* like they came from a heads-heavy coin. Trials 1 and 4 (5 and 4
> heads) feel more balanced. Trial 5 (7 heads) is somewhere in between.
> That intuition — softly assigning trials to coins — is exactly what the
> EM algorithm formalizes.

**Visualization 1 — `<BinomialMixture>`** (full width):

Two side-by-side bar charts of the binomial PMFs $\text{Bin}(10, \theta_A)$
and $\text{Bin}(10, \theta_B)$ over $k \in \{0, 1, \ldots, 10\}$. Below them,
a third chart showing the mixture
$\frac{1}{2}\text{Bin}(10, \theta_A) + \frac{1}{2}\text{Bin}(10, \theta_B)$.

Two sliders let the user adjust $\theta_A$ and $\theta_B$ from 0.05 to 0.95.
A horizontal strip below the mixture plot marks the actual observed values
$x_i = \{5, 9, 8, 4, 7\}$ as small dots, color-mixed by which coin's PMF is
larger at that $k$.

The pedagogical purpose: when $\theta_A$ and $\theta_B$ are far apart, the
two PMFs barely overlap — coins are easy to tell apart. When they're close,
the mixture is nearly indistinguishable from a single binomial — separation
is impossible.

**Callout — "Why this is hard"** (`type=info`):

> If we knew which coin made each trial, this problem would be trivial.
> Group trials by coin, divide heads by total flips, done. But we don't.
> So we're stuck with a likelihood that *mixes* both coins together at every
> trial — and mixtures don't have closed-form maxima.

---

### Section 3 — Complete Data vs Incomplete Data

**Length**: ~600 words. This is the formal foundation.

**Prose**:

> Let's give names to the two versions of this problem.
>
> #### The complete-data problem
>
> If we *could* see $Z_i$, we'd have **complete data** $(x_i, z_i)_{i=1}^{5}$.
> The likelihood factorizes beautifully:
>
> $$L_c(\theta \mid x, z) \;=\; \prod_{i=1}^{5} \binom{10}{x_i} \, \theta_{z_i}^{x_i} (1 - \theta_{z_i})^{10 - x_i}$$
>
> The complete-data **log-likelihood** (dropping the $\binom{10}{x_i}$ since
> it doesn't depend on $\theta$) is:
>
> $$\ell_c(\theta \mid x, z) \;=\; \sum_{i=1}^{5} \Big[\, x_i \log \theta_{z_i} + (10 - x_i) \log(1 - \theta_{z_i}) \,\Big]$$
>
> A cleaner way to write that, using indicator functions, is:
>
> $$\ell_c(\theta \mid x, z) \;=\; \sum_{i=1}^{5} \mathbf{1}[z_i = A] \big( x_i \log \theta_A + (10-x_i)\log(1-\theta_A) \big) + \mathbf{1}[z_i = B] \big( x_i \log \theta_B + (10-x_i)\log(1-\theta_B) \big)$$
>
> Maximizing this over $\theta$ is trivial. The MLE just groups trials by
> their assigned coin and takes the heads fraction:
>
> $$\hat\theta_A \;=\; \frac{\sum_{i:\, z_i = A} x_i}{10 \cdot \#\{i : z_i = A\}}, \qquad \hat\theta_B \;=\; \frac{\sum_{i:\, z_i = B} x_i}{10 \cdot \#\{i : z_i = B\}}$$
>
> #### The incomplete-data problem
>
> But we don't observe $z_i$. To get the likelihood of $x_i$ alone we have
> to **marginalize** over the unknown $Z_i$:
>
> $$P(x_i \mid \theta) \;=\; \sum_{z \in \{A,B\}} P(Z_i = z) \cdot P(x_i \mid Z_i = z, \theta) \;=\; \frac{1}{2}\binom{10}{x_i}\theta_A^{x_i}(1-\theta_A)^{10-x_i} + \frac{1}{2}\binom{10}{x_i}\theta_B^{x_i}(1-\theta_B)^{10-x_i}$$
>
> The full incomplete-data likelihood is the product over trials:
>
> $$L(\theta \mid x) \;=\; \prod_{i=1}^{5} \frac{1}{2}\binom{10}{x_i}\Big[\, \theta_A^{x_i}(1-\theta_A)^{10-x_i} + \theta_B^{x_i}(1-\theta_B)^{10-x_i} \,\Big]$$
>
> And the log-likelihood — the function we **actually want to maximize**:
>
> $$\ell(\theta \mid x) \;=\; \sum_{i=1}^{5} \log\Big[\, \theta_A^{x_i}(1-\theta_A)^{10-x_i} + \theta_B^{x_i}(1-\theta_B)^{10-x_i} \,\Big] \;+\; \text{const}$$

**Callout — "The obstacle"** (`type=warning`):

> Notice the structure. The complete-data log-likelihood is a sum of logs
> (clean, separable, easy to differentiate). The incomplete-data
> log-likelihood is a sum of **logs of sums**. When you take
> $\partial / \partial \theta_A$, the derivative drags every $\theta_A$ out
> of a denominator that mixes both parameters together. There is no closed
> form for the MLE.
>
> This is *the* fundamental obstacle. Everything EM does is a workaround for
> this one fact.

**Visualization 2 — `<LikelihoodSurface>`** (full width, ~600px tall):

A 2D contour plot of $\ell(\theta_A, \theta_B \mid x)$ over
$(0.05, 0.95) \times (0.05, 0.95)$ with ~12 contour levels. Compute the
log-likelihood numerically on a 200×200 grid using `observedLogLikelihood`.

Key features:
- Two **symmetric global maxima** near $(0.797, 0.520)$ and $(0.520, 0.797)$
  (the model can't distinguish coin labels — that's a real property worth
  discussing).
- A **saddle ridge** along the diagonal $\theta_A = \theta_B$.
- Mark the two maxima with stars.

Toggle button: **"Show complete-data version"** — recomputes assuming a
specific assignment $z = (B, A, A, B, A)$ (a guess) and shows the
single-peaked, smooth, easy-to-maximize $\ell_c$. Side-by-side this tells
the entire pedagogical story: complete-data surface is a clean bowl;
incomplete-data surface is two-peaked, ridged, and nasty.

Hover any $(\theta_A, \theta_B)$ point and a tooltip displays the numeric
log-likelihood there.

---

### Section 4 — The Key Idea

**Length**: ~400 words.

**Prose**:

> Here is the trick.
>
> We can't see $Z_i$. But what if we use our *current best guess* of $\theta$
> to compute the **probability** that each trial came from each coin? That
> gives us a soft, probabilistic assignment of trials to coins. Then we can
> compute an "expected" complete-data log-likelihood — pretending we have
> weighted versions of the missing $z_i$ — and maximize *that* instead.
>
> Then we use the new $\theta$ to refine our soft assignments. Repeat. The
> remarkable theorem (we'll prove it later) is: **this procedure climbs the
> true, intractable, incomplete-data log-likelihood, monotonically.** We
> never have to differentiate the messy log-of-sum directly. We just
> alternate two easy steps.
>
> Formally, the EM algorithm produces a sequence
> $\theta^{(0)}, \theta^{(1)}, \theta^{(2)}, \ldots$ where each step alternates:
>
> **E-step.** Given the current $\theta^{(t)}$, compute the conditional
> distribution of the missing data:
>
> $$k(z \mid x, \theta^{(t)}) \;:=\; P(Z = z \mid x, \theta^{(t)}) \;=\; \frac{P(x, z \mid \theta^{(t)})}{P(x \mid \theta^{(t)})}$$
>
> Use it to form the function (of $\theta$):
>
> $$\boxed{\; Q(\theta \mid \theta^{(t)}) \;:=\; \mathbb{E}_{Z \,\sim\, k(\cdot \mid x, \theta^{(t)})} \big[\, \ell_c(\theta \mid x, Z) \,\big] \;}$$
>
> **M-step.** Maximize $Q$ over $\theta$:
>
> $$\boxed{\; \theta^{(t+1)} \;:=\; \arg\max_\theta \; Q(\theta \mid \theta^{(t)}) \;}$$

**Callout — "Read this twice"** (`type=tip`):

> The function $Q(\theta \mid \theta^{(t)})$ uses $\theta$ in **two distinct
> roles**. The $\theta^{(t)}$ on the right of the bar is *fixed* — it's just
> our current guess used to compute the expectation. The $\theta$ inside the
> log-likelihood $\ell_c(\theta \mid x, Z)$ is the **variable** we'll
> maximize over. They are not the same. Mixing them up is the #1 source of
> confusion when learning EM.

**Visualization 3 — `<ELBODiagram>`** (medium width):

A 1D illustration. Plot the true log-likelihood $\ell(\theta)$ as a curve
(sliced through one parameter, say $\theta_A$ with $\theta_B$ fixed at its
current value). At point $\theta^{(t)}$, plot the function
$Q(\theta \mid \theta^{(t)}) - H(\theta^{(t)} \mid \theta^{(t)})$ as a
**lower bound** curve that touches $\ell(\theta)$ exactly at $\theta^{(t)}$.

Buttons: "Step E", "Step M", "Reset". Watch the lower bound get maximized
(M-step → $\theta^{(t+1)}$), then a *new* lower bound replace it (E-step at
the new point), tangent at the new point. The reader sees the
"majorize-then-maximize" pattern visually. This is the canonical EM picture
and is worth real implementation effort.

---

### Section 5 — The E-Step, Concretely

**Length**: ~550 words. This is where things get real.

**Prose**:

> Let's compute the E-step for the two coins. We need, for each trial, the
> conditional probability that Coin A was used given the observed heads
> count. By Bayes' rule:
>
> $$\gamma_i^A \;:=\; P(Z_i = A \mid x_i, \theta^{(t)}) \;=\; \frac{P(Z_i = A) \cdot P(x_i \mid Z_i = A, \theta^{(t)})}{P(x_i \mid \theta^{(t)})}$$
>
> Plug in $P(Z_i = A) = \tfrac{1}{2}$ and the binomial PMFs. The
> $\binom{10}{x_i}$ factor and the $\tfrac{1}{2}$ both appear in numerator
> and denominator, so they cancel:

> $$\boxed{\;\; \gamma_i^A \;=\; \frac{(\theta_A^{(t)})^{x_i}\,(1-\theta_A^{(t)})^{10-x_i}}{(\theta_A^{(t)})^{x_i}\,(1-\theta_A^{(t)})^{10-x_i} \;+\; (\theta_B^{(t)})^{x_i}\,(1-\theta_B^{(t)})^{10-x_i}} \;\;}$$
>
> and $\gamma_i^B = 1 - \gamma_i^A$. These quantities are called
> **responsibilities**: $\gamma_i^A$ is the probability that *Coin A is
> responsible for trial $i$*, given everything we currently believe.

> Now plug them into the expected complete-data log-likelihood. Recall
> $\ell_c$ uses indicator functions $\mathbf{1}[Z_i = A]$. Taking expectation
> under $Z \sim k(\cdot \mid x, \theta^{(t)})$ replaces each indicator with
> its expectation, which is exactly the responsibility:
>
> $$\mathbb{E}\big[ \mathbf{1}[Z_i = A] \big] \;=\; P(Z_i = A \mid x_i, \theta^{(t)}) \;=\; \gamma_i^A$$
>
> So:
>
> $$Q(\theta \mid \theta^{(t)}) \;=\; \sum_{i=1}^{5} \Big[\, \gamma_i^A \big( x_i \log \theta_A + (10 - x_i) \log(1 - \theta_A) \big) \;+\; \gamma_i^B \big( x_i \log \theta_B + (10 - x_i) \log(1 - \theta_B) \big) \,\Big]$$
>
> That's the E-step. We have $Q$ as an explicit function of $\theta$.

**Worked numerical example block** (styled box, monospace numerics):

> Initialize $\theta^{(0)} = (0.6, 0.5)$. Compute $\gamma_1^A$ for trial 1
> ($x_1 = 5$):
>
> $$\gamma_1^A \;=\; \frac{0.6^5 \cdot 0.4^5}{0.6^5 \cdot 0.4^5 \;+\; 0.5^5 \cdot 0.5^5} \;=\; \frac{0.0007963}{0.0007963 + 0.0009766} \;\approx\; 0.4491$$
>
> So given the initial guess, trial 1 is *slightly more* likely to have
> come from Coin B. Repeat for all five trials:

| $i$ | $x_i$ | $\gamma_i^A$ | $\gamma_i^B$ |
|:---:|:-----:|:------------:|:------------:|
| 1 | 5 | 0.4491 | 0.5509 |
| 2 | 9 | 0.8050 | 0.1950 |
| 3 | 8 | 0.7335 | 0.2665 |
| 4 | 4 | 0.3522 | 0.6478 |
| 5 | 7 | 0.6472 | 0.3528 |

(These are real, computed values. The implementing agent should verify
these in `algorithm.test.ts`.)

**Visualization 4 — `<ResponsibilityCalculator>`** (full width):

The reader controls $\theta_A$ and $\theta_B$ via two sliders (range
0.05–0.95, step 0.01). Beneath, a panel of 5 horizontal stacked bars,
one per trial. Each bar is filled with `--coin-a` (left, width $\propto
\gamma_i^A$) and `--coin-b` (right, width $\propto \gamma_i^B$).
Numerical values to 4 decimals show on the right of each bar.

A small sparkline above shows the relative likelihood ratio
$\log(\gamma_i^A / \gamma_i^B)$ for an at-a-glance view. As sliders move,
bars update smoothly with D3 transitions.

The default state shows $(\theta_A, \theta_B) = (0.6, 0.5)$ with the
exact responsibilities from the worked example above — so the reader can
verify the numbers in the table by reading them off the bars.

---

### Section 6 — The M-Step, Concretely

**Length**: ~400 words.

**Prose**:

> Now maximize $Q(\theta \mid \theta^{(t)})$ over $\theta$. Notice the
> function splits cleanly into a $\theta_A$-only part and a $\theta_B$-only
> part:
>
> $$Q(\theta \mid \theta^{(t)}) \;=\; \underbrace{\sum_{i} \gamma_i^A \big[ x_i \log \theta_A + (10-x_i)\log(1-\theta_A) \big]}_{\text{depends only on } \theta_A} \;+\; \underbrace{\sum_{i} \gamma_i^B \big[ x_i \log \theta_B + (10-x_i)\log(1-\theta_B) \big]}_{\text{depends only on } \theta_B}$$
>
> So we can maximize over $\theta_A$ and $\theta_B$ independently.
> Differentiate the $\theta_A$ piece with respect to $\theta_A$, set to zero:
>
> $$\sum_{i=1}^{5} \gamma_i^A \left[ \frac{x_i}{\theta_A} \;-\; \frac{10 - x_i}{1 - \theta_A} \right] \;=\; 0$$
>
> Multiply through by $\theta_A(1 - \theta_A)$:
>
> $$(1 - \theta_A) \sum_{i} \gamma_i^A \, x_i \;=\; \theta_A \sum_{i} \gamma_i^A \, (10 - x_i)$$
>
> Expand and collect:
>
> $$\sum_{i} \gamma_i^A \, x_i \;=\; \theta_A \sum_{i} \gamma_i^A \cdot 10$$
>
> Solving:
>
> $$\boxed{\;\; \theta_A^{(t+1)} \;=\; \frac{\displaystyle \sum_{i=1}^{5} \gamma_i^A \cdot x_i}{\displaystyle \sum_{i=1}^{5} \gamma_i^A \cdot 10} \;\;}$$
>
> By the same calculation:
>
> $$\theta_B^{(t+1)} \;=\; \frac{\displaystyle \sum_{i=1}^{5} \gamma_i^B \cdot x_i}{\displaystyle \sum_{i=1}^{5} \gamma_i^B \cdot 10}$$

**Callout — "Interpret the formula"** (`type=tip`):

> The new $\theta_A$ is just the **weighted fraction of heads**, where each
> trial is weighted by how responsible Coin A is for it. Trials Coin A
> "owns" (high $\gamma_i^A$) contribute fully; trials Coin A barely owns
> contribute little. This is exactly the same as the complete-data MLE —
> only with **soft counts** instead of hard ones.

**Worked numerical example continuation** (continuing from §5):

> Plug in the responsibilities from §5 with $x = (5, 9, 8, 4, 7)$:
>
> $$\sum_i \gamma_i^A x_i \;=\; 0.4491(5) + 0.8050(9) + 0.7335(8) + 0.3522(4) + 0.6472(7) \;=\; 21.295$$
>
> $$\sum_i \gamma_i^A \cdot 10 \;=\; 10 \cdot (0.4491 + 0.8050 + 0.7335 + 0.3522 + 0.6472) \;=\; 29.870$$
>
> $$\theta_A^{(1)} \;=\; \frac{21.295}{29.870} \;\approx\; 0.7130$$
>
> Similarly $\theta_B^{(1)} \approx 0.5813$. **One iteration moved
> $(0.6, 0.5)$ to $(0.7130, 0.5813)$ — already separating toward the true
> answer.** After 10 iterations the algorithm settles at
> $(0.7967, 0.5197)$.

A small inline table shows the parameter trajectory for the first 10 iters:

| $t$ | $\theta_A^{(t)}$ | $\theta_B^{(t)}$ | $\ell(\theta^{(t)} \mid x)$ |
|:---:|:----------------:|:----------------:|:---------------------------:|
| 0 | 0.6000 | 0.5000 | −33.094 |
| 1 | 0.7130 | 0.5813 | −31.859 |
| 2 | 0.7453 | 0.5693 | −31.723 |
| 3 | 0.7681 | 0.5495 | −31.628 |
| 4 | 0.7832 | 0.5346 | −31.585 |
| 5 | 0.7911 | 0.5263 | −31.573 |
| 6 | 0.7945 | 0.5224 | −31.571 |
| 7 | 0.7959 | 0.5207 | −31.570 |
| 8 | 0.7965 | 0.5200 | −31.570 |
| 9 | 0.7967 | 0.5198 | −31.570 |
| 10 | 0.7967 | 0.5197 | −31.570 |

(Verify: implementing agent should reproduce this table from
`runUntilConvergence(0.6, 0.5)` — these numbers are pre-computed and known
correct.)

---

### Section 7 — The Full Algorithm

**Length**: ~250 words + the centerpiece simulator.

**Prose** (introduces the full pseudocode):

> Putting both steps together, the algorithm is:

Boxed pseudocode block (monospace, syntax-highlighted):

```
EM Algorithm — Two Coins

Input:  observed data x = (x_1, ..., x_5)
        initial parameters θ^(0) = (θ_A^(0), θ_B^(0))
        tolerance ε

Initialize:  t = 0

Repeat:
  // E-step
  for i = 1 to 5:
      a := (θ_A^(t))^x_i · (1 − θ_A^(t))^(10−x_i)
      b := (θ_B^(t))^x_i · (1 − θ_B^(t))^(10−x_i)
      γ_i^A := a / (a + b)
      γ_i^B := 1 − γ_i^A

  // M-step
  θ_A^(t+1) := ( Σ_i γ_i^A · x_i ) / ( 10 · Σ_i γ_i^A )
  θ_B^(t+1) := ( Σ_i γ_i^B · x_i ) / ( 10 · Σ_i γ_i^B )

  // Convergence check
  if ‖ θ^(t+1) − θ^(t) ‖ < ε:  break
  t := t + 1

Return θ^(t+1)
```

**The Centerpiece — `<EMSimulator>`** (full-width, ≥720px tall):

This is the most important interactive element on the page. Layout:

```
┌────────────────────────────────────────────────────────────────────┐
│  CONTROLS  (sticky bar at top of component)                         │
│  ─────────────────────────────────────────────────────────────────  │
│  θ_A^(0):  ◉━━━━━━━ 0.60   θ_B^(0):  ━◉━━━━━ 0.50                  │
│  [ Step E ] [ Step M ] [ Step Both ] [ ▶ Play ] [ ⟲ Reset ]         │
│  Iteration: 3   ‖Δθ‖ = 0.0228   ℓ(θ|x) = −31.628                    │
│                                                                     │
│  Quick presets: [Symmetric (0.5,0.5)] [Asymmetric (0.8,0.2)]        │
│                 [Adversarial (0.3,0.7)] [Random]                    │
└────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────┬─────────────────────────────────────┐
│  RESPONSIBILITIES            │  TRAJECTORY ON LIKELIHOOD SURFACE   │
│  (5 horizontal stacked bars) │  (contour plot from §3 with the     │
│  Updates each step           │   sequence θ^(0), θ^(1), ...        │
│                              │   drawn as a connected polyline)    │
├──────────────────────────────┼─────────────────────────────────────┤
│  PARAMETERS OVER ITERATIONS  │  LOG-LIKELIHOOD OVER ITERATIONS     │
│  Two lines (θ_A, θ_B)        │  Single line, monotonic             │
│  vs. iteration count         │  Annotated: "this is monotonic by   │
│                              │   Theorem 7.2.20"                   │
└──────────────────────────────┴─────────────────────────────────────┘
```

Behavior details:

- **Step E**: highlights the responsibility panel, animates bars to new
  values. The trajectory and parameter charts do *not* update yet
  (E-step doesn't change $\theta$).
- **Step M**: animates $\theta_A, \theta_B$ to new values. Trajectory adds
  a new vertex; parameter chart and log-likelihood chart each gain a point.
- **Step Both**: does E then M with a 400ms beat between them.
- **Play**: auto-steps every 600ms. Stops automatically when
  $\|\theta^{(t+1)} - \theta^{(t)}\| < 10^{-6}$ or after 100 iterations.
  Speed slider hidden in a "settings" gear icon.
- **Reset**: restores the user's slider values as $\theta^{(0)}$, clears all
  history.

State management: keep an array `EMState[]` (history) in the component.
Resetting truncates to length 1; stepping appends.

**Preset buttons explained** (each, on click, sets sliders + resets):
- **Symmetric (0.5, 0.5)**: EM is *stuck at the saddle*. After E-step, all
  $\gamma_i^A = 0.5$. After M-step, $\theta_A^{(1)} = \theta_B^{(1)} = $ the
  overall heads fraction (33/50 = 0.66). It then stays put forever. This
  is a real and important phenomenon.
- **Asymmetric (0.8, 0.2)**: converges quickly to $\approx (0.797, 0.520)$.
- **Adversarial (0.3, 0.7)**: converges to the *label-swapped* solution
  $\approx (0.520, 0.797)$. Both are global maxima.
- **Random**: pick $\theta_A, \theta_B$ uniformly in $(0.1, 0.9)$.

---

### Section 8 — Why It Works (Convergence)

**Length**: ~600 words. The proof in full. **Do not skip steps** — the
reader explicitly wants to see the math.

**Prose**:

> The remarkable fact about EM is:
>
> **Theorem (Monotonicity of EM).** For every $t$,
> $$\ell(\theta^{(t+1)} \mid x) \;\geq\; \ell(\theta^{(t)} \mid x).$$
> Equality holds if and only if
> $Q(\theta^{(t+1)} \mid \theta^{(t)}) = Q(\theta^{(t)} \mid \theta^{(t)})$
> (i.e., the M-step makes no progress).
>
> **Proof.** Use the conditional density of the missing data:
>
> $$k(z \mid x, \theta) \;:=\; P(Z = z \mid x, \theta) \;=\; \frac{P(x, z \mid \theta)}{P(x \mid \theta)}$$
>
> Take the log on both sides of the rearranged identity
> $P(x, z \mid \theta) = P(x \mid \theta) \cdot k(z \mid x, \theta)$:
>
> $$\log P(x, z \mid \theta) \;=\; \log P(x \mid \theta) \;+\; \log k(z \mid x, \theta)$$
>
> Equivalently, $\ell_c(\theta \mid x, z) = \ell(\theta \mid x) + \log k(z \mid x, \theta)$.
> Solving for $\ell(\theta \mid x)$ (which doesn't depend on $z$):
>
> $$\ell(\theta \mid x) \;=\; \ell_c(\theta \mid x, z) \;-\; \log k(z \mid x, \theta)$$
>
> Now take the expectation of *both sides* under $z \sim k(\cdot \mid x, \theta^{(t)})$.
> The left side is unchanged (it doesn't depend on $z$):
>
> $$\ell(\theta \mid x) \;=\; \underbrace{\mathbb{E}_{Z \sim k(\cdot \mid x, \theta^{(t)})} [\ell_c(\theta \mid x, Z)]}_{=\, Q(\theta \mid \theta^{(t)})} \;-\; \underbrace{\mathbb{E}_{Z \sim k(\cdot \mid x, \theta^{(t)})} [\log k(Z \mid x, \theta)]}_{=:\, H(\theta \mid \theta^{(t)})}$$
>
> So we have the **decomposition identity**:
>
> $$\ell(\theta \mid x) \;=\; Q(\theta \mid \theta^{(t)}) \;-\; H(\theta \mid \theta^{(t)})$$
>
> Therefore the change in log-likelihood between iterations is:
>
> $$\ell(\theta^{(t+1)} \mid x) \;-\; \ell(\theta^{(t)} \mid x) \;=\; \underbrace{\big[ Q(\theta^{(t+1)} \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)}) \big]}_{\text{(A)}} \;-\; \underbrace{\big[ H(\theta^{(t+1)} \mid \theta^{(t)}) - H(\theta^{(t)} \mid \theta^{(t)}) \big]}_{\text{(B)}}$$
>
> We show **(A)** is non-negative and **(B)** is non-positive — making the
> total non-negative.
>
> **Term (A) ≥ 0.** This is immediate from the M-step's definition:
> $\theta^{(t+1)}$ is *defined* to maximize $Q(\theta \mid \theta^{(t)})$ over
> $\theta$. So in particular $Q(\theta^{(t+1)} \mid \theta^{(t)}) \geq Q(\theta^{(t)} \mid \theta^{(t)})$.
>
> **Term (B) ≤ 0.** This is **Gibbs' inequality** (equivalently,
> non-negativity of KL divergence). For any two distributions $p, q$ on the
> same space,
> $$\mathbb{E}_{Z \sim p}[\log p(Z)] \;\geq\; \mathbb{E}_{Z \sim p}[\log q(Z)]$$
> with equality iff $p = q$. Apply this with
> $p = k(\cdot \mid x, \theta^{(t)})$ and $q = k(\cdot \mid x, \theta^{(t+1)})$:
>
> $$H(\theta^{(t)} \mid \theta^{(t)}) \;=\; \mathbb{E}_{Z \sim k(\cdot \mid x, \theta^{(t)})}[\log k(Z \mid x, \theta^{(t)})] \;\geq\; \mathbb{E}_{Z \sim k(\cdot \mid x, \theta^{(t)})}[\log k(Z \mid x, \theta^{(t+1)})] \;=\; H(\theta^{(t+1)} \mid \theta^{(t)})$$
>
> So $H(\theta^{(t+1)} \mid \theta^{(t)}) - H(\theta^{(t)} \mid \theta^{(t)}) \leq 0$,
> i.e. term (B) is non-positive. Equivalently:
>
> $$H(\theta^{(t)} \mid \theta^{(t)}) - H(\theta^{(t+1)} \mid \theta^{(t)}) \;=\; \mathrm{KL}\big(k(\cdot \mid x, \theta^{(t)}) \,\big\|\, k(\cdot \mid x, \theta^{(t+1)})\big) \;\geq\; 0$$
>
> Combining:
>
> $$\ell(\theta^{(t+1)} \mid x) - \ell(\theta^{(t)} \mid x) \;=\; \underbrace{[Q\text{-gap}]}_{\geq 0} \;+\; \underbrace{\mathrm{KL}\big(k(\cdot \mid x, \theta^{(t)}) \big\| k(\cdot \mid x, \theta^{(t+1)})\big)}_{\geq 0} \;\geq\; 0. \quad \blacksquare$$

Wrap the proof in a `<ProofToggle>` collapsed by default with the trigger
"Show full proof of monotonicity (recommended)". Once expanded, render
exactly as above.

**Visualization 5 — `<MonotonicityDemo>`** (medium width):

A live bar chart synced to the `<EMSimulator>` (via shared event bus). For
each iteration $t$ that has been computed, two stacked bars:
- Lower segment (in `--amber`): the Q-gap, $Q(\theta^{(t+1)} \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)})$
- Upper segment (in `--sage`): the KL term, $\mathrm{KL}\big(k(\cdot \mid x, \theta^{(t)}) \| k(\cdot \mid x, \theta^{(t+1)})\big)$

Total bar height = log-likelihood improvement that step. Both segments are
always non-negative — the reader sees the proof in action.

**Caveat callout — "What EM does NOT guarantee"** (`type=warning`):

> EM is **monotonic**, but it is **not guaranteed to find the global
> maximum**. It can get stuck:
>
> - **Local maxima.** Bad initializations may converge to suboptimal points.
>   In our two-coins problem the surface has only two (symmetric) global
>   maxima and a single saddle, so this is mostly OK. In higher-dimensional
>   problems (e.g. Gaussian mixture models) local maxima are routine.
> - **Saddle points.** Try the "Symmetric" preset in the simulator. EM
>   stays at the saddle forever, because at $\theta_A = \theta_B$ all
>   responsibilities are 0.5 and the M-step can't break the symmetry.
> - **Slow convergence.** Near saddle points or in flat regions EM crawls.
>
> Standard practice: run from many random initializations and keep the run
> with the highest final $\ell(\theta \mid x)$.

---

### Section 9 — Pitfalls and Practical Notes

**Length**: ~350 words.

Cover, in short prose paragraphs:

1. **Label switching.** $(\theta_A, \theta_B) = (0.80, 0.52)$ and $(0.52,
   0.80)$ are equivalent — the model can't tell coin labels apart. This
   matters when comparing or averaging EM runs.

2. **Initialization sensitivity.** The "Symmetric" preset demonstrates
   this. Random restarts help.

3. **What if priors aren't 50/50?** Generalize: let $\pi_A = P(Z_i = A)$
   also be a parameter. The responsibility formula picks up the prior:
   $$\gamma_i^A \;=\; \frac{\pi_A \cdot \theta_A^{x_i}(1-\theta_A)^{10-x_i}}{\pi_A \cdot \theta_A^{x_i}(1-\theta_A)^{10-x_i} + \pi_B \cdot \theta_B^{x_i}(1-\theta_B)^{10-x_i}}$$
   and the M-step gains an update $\pi_A^{(t+1)} = \tfrac{1}{n}\sum_i \gamma_i^A$.
   This is the standard Bernoulli mixture model.

4. **Stopping criteria.** Either parameter change
   $\|\theta^{(t+1)} - \theta^{(t)}\| < \varepsilon$ or log-likelihood
   change $\ell(\theta^{(t+1)} \mid x) - \ell(\theta^{(t)} \mid x) < \varepsilon$.
   The latter is more principled (it's the quantity we're actually
   trying to maximize).

5. **Numerical stability.** When $\theta$ approaches 0 or 1, computing
   $\theta^{x_i}(1-\theta)^{10-x_i}$ underflows. Always work in log space
   and use the log-sum-exp trick. Implementation detail in §6.

**Visualization 6 — `<MultiRestartGallery>`** (full width):

Run EM from 30 random initializations. Plot all 30 trajectories
simultaneously on the §3 likelihood surface contour map. Color each
trajectory by which mode it converges to (the two global maxima get
distinct colors; saddle-stuck runs get a third). A small histogram on the
side counts how many runs landed where. The visual punchline:
most random initializations land at one of the two global maxima, but a
small fraction get stuck.

A button "Re-roll" generates a fresh batch.

---

### Section 10 — Summary

**Length**: ~150 words + a "what you've learned" card.

A clean closing block:

> #### What you now know
>
> - The EM algorithm finds maximum-likelihood estimates when there's
>   missing or latent data.
> - It alternates **E-step** (compute the conditional distribution of the
>   missing data, form $Q$) with **M-step** (maximize $Q$).
> - In the two-coins problem, the E-step computes responsibilities; the
>   M-step computes a weighted heads-fraction.
> - The observed log-likelihood is **monotonically non-decreasing** under
>   EM (proved via the $Q$-$H$ decomposition and Gibbs' inequality).
> - EM converges to a *local* maximum; multiple random restarts help.
> - The two-coins setup is a **Bernoulli mixture model** — the simplest
>   instance of a much larger family. Gaussian mixture models, hidden
>   Markov models (forward–backward), latent Dirichlet allocation, and
>   k-means (in the limit) are all EM in disguise.

Followed by a "where next" panel:
- Bishop, *Pattern Recognition and Machine Learning*, ch. 9
- Murphy, *Probabilistic Machine Learning*, ch. 8
- Dempster, Laird & Rubin (1977), the original EM paper
- Do & Batzoglou (2008), *Nat. Biotech.* — source of this example

---

## 5. Algorithm Implementation (`src/em/algorithm.ts`)

Numerically stable. Use log-sum-exp throughout.

```ts
import { TRIALS, FLIPS_PER_TRIAL, PRIOR } from './data';

export interface EMState {
  thetaA: number;
  thetaB: number;
  iteration: number;
  responsibilities: { gammaA: number; gammaB: number }[];
  observedLogLikelihood: number;
}

const EPS = 1e-12;
const clamp = (t: number) => Math.max(EPS, Math.min(1 - EPS, t));

/** log P(x_i | Z_i = c, θ_c) up to the binomial coefficient (constant in θ). */
function trialLogWeight(theta: number, heads: number, tails: number): number {
  const t = clamp(theta);
  return heads * Math.log(t) + tails * Math.log(1 - t);
}

/** E-step: returns responsibilities for each trial. */
export function eStep(thetaA: number, thetaB: number) {
  return TRIALS.map(({ heads, tails }) => {
    const lA = Math.log(PRIOR)     + trialLogWeight(thetaA, heads, tails);
    const lB = Math.log(1 - PRIOR) + trialLogWeight(thetaB, heads, tails);
    const m = Math.max(lA, lB);
    const a = Math.exp(lA - m);
    const b = Math.exp(lB - m);
    const gammaA = a / (a + b);
    return { gammaA, gammaB: 1 - gammaA };
  });
}

/** M-step: weighted MLE for θ_A and θ_B. */
export function mStep(R: { gammaA: number; gammaB: number }[]): { thetaA: number; thetaB: number } {
  let nA = 0, dA = 0, nB = 0, dB = 0;
  R.forEach(({ gammaA, gammaB }, i) => {
    const { heads } = TRIALS[i];
    nA += gammaA * heads;
    dA += gammaA * FLIPS_PER_TRIAL;
    nB += gammaB * heads;
    dB += gammaB * FLIPS_PER_TRIAL;
  });
  return { thetaA: nA / dA, thetaB: nB / dB };
}

/** Observed-data log-likelihood ℓ(θ | x) (drops constant binomial coefficients). */
export function observedLogLikelihood(thetaA: number, thetaB: number): number {
  return TRIALS.reduce((acc, { heads, tails }) => {
    const lA = Math.log(PRIOR)     + trialLogWeight(thetaA, heads, tails);
    const lB = Math.log(1 - PRIOR) + trialLogWeight(thetaB, heads, tails);
    const m = Math.max(lA, lB);
    return acc + m + Math.log(Math.exp(lA - m) + Math.exp(lB - m));
  }, 0);
}

export function initialState(thetaA0: number, thetaB0: number): EMState {
  const R = eStep(thetaA0, thetaB0);
  return {
    thetaA: thetaA0, thetaB: thetaB0,
    iteration: 0,
    responsibilities: R,
    observedLogLikelihood: observedLogLikelihood(thetaA0, thetaB0),
  };
}

export function runEMStep(s: EMState): EMState {
  const R = eStep(s.thetaA, s.thetaB);
  const { thetaA, thetaB } = mStep(R);
  return {
    thetaA, thetaB,
    iteration: s.iteration + 1,
    responsibilities: eStep(thetaA, thetaB), // refresh γ for the new θ
    observedLogLikelihood: observedLogLikelihood(thetaA, thetaB),
  };
}

export function runUntilConvergence(thetaA0: number, thetaB0: number,
                                    maxIter = 200, tol = 1e-8): EMState[] {
  const history = [initialState(thetaA0, thetaB0)];
  for (let t = 0; t < maxIter; t++) {
    const last = history[history.length - 1];
    const next = runEMStep(last);
    history.push(next);
    if (Math.abs(next.thetaA - last.thetaA) + Math.abs(next.thetaB - last.thetaB) < tol) break;
  }
  return history;
}
```

**Test cases** (in `algorithm.test.ts`):

- `runUntilConvergence(0.6, 0.5)` produces, at iteration 10,
  `thetaA ≈ 0.7967`, `thetaB ≈ 0.5197`, `observedLogLikelihood ≈ -31.5702`.
- `runUntilConvergence(0.5, 0.5)` produces no progress on $\theta_A \neq
  \theta_B$ (saddle stationarity).
- `observedLogLikelihood` is non-decreasing across iterations of any run.

---

## 6. Component Catalog (full list)

### Layout / chrome
- `<NavigationSidebar>` — sticky left TOC, desktop ≥1024px. Highlights the
  current section via `IntersectionObserver`. Click to smooth-scroll.
- `<ProgressBar>` — top-fixed bar, all viewports. Width = scroll progress.
- `<Footer>` — credits, source paper citations.

### Prose primitives
- `<MathInline>` / `<MathBlock>` — wrappers around KaTeX, handle errors.
- `<Callout type="info|tip|warning|proof">` — colored aside box with icon.
- `<ProofToggle title>` — collapsible, animated height. Uses `<details>`
  under the hood for accessibility.
- `<DataTable>` — styled HTML table. Hover-highlight rows.
- `<Pseudocode>` — monospace block, faux syntax highlighting for keywords.

### Visualizations (each gets its own `viz/*.ts` file)
- `<BinomialMixture>` — §2
- `<LikelihoodSurface>` — §3 (also reused as a backdrop in §7 and §9)
- `<ELBODiagram>` — §4
- `<ResponsibilityCalculator>` — §5
- `<EMSimulator>` — §7 (centerpiece; composes responsibility bars,
  trajectory plot, parameter chart, log-likelihood chart)
- `<MonotonicityDemo>` — §8
- `<MultiRestartGallery>` — §9

Each viz component exposes a `mount(container, props)` API and an
`update(props)` API for re-renders. They communicate (when needed) via a
small `EventBus` singleton — e.g. `EMSimulator` emits `step` events that
`MonotonicityDemo` listens to.

---

## 7. Page-Level UX

- Sticky sidebar lists §1–§10. Clicking jumps via smooth scroll. Active
  section highlighted (IntersectionObserver, threshold 0.4).
- Mobile: sidebar collapses into a hamburger menu in the top-right of the
  fixed top bar. Progress bar always visible.
- Keyboard: ← / → cycle sections; Space steps the active EM simulator (if
  one is in the viewport).
- All visualizations honor `prefers-reduced-motion`: instant transitions
  instead of animated ones.
- All visualizations are keyboard-accessible: sliders use `<input
  type="range">`, buttons are real `<button>`s, focus rings are visible.
- Color: all paired colors (Coin A red, Coin B blue) maintain ≥3:1
  contrast on the paper background. Colorblind-safe palette (red and blue
  stay distinguishable for deuteranopia).

---

## 8. Acceptance Criteria

A learner who has worked through this page should be able to, on a blank
sheet of paper:

1. Explain in their own words why incomplete-data MLE is hard but
   complete-data MLE is easy. Reference the *log of a sum* problem.
2. Write down the responsibility formula
   $\gamma_i^A = \dfrac{\theta_A^{x_i}(1-\theta_A)^{10-x_i}}{\theta_A^{x_i}(1-\theta_A)^{10-x_i} + \theta_B^{x_i}(1-\theta_B)^{10-x_i}}$
   from memory.
3. Write down the M-step update
   $\theta_A^{(t+1)} = \dfrac{\sum_i \gamma_i^A x_i}{10 \sum_i \gamma_i^A}$
   and explain why it's a *weighted MLE*.
4. State the monotonicity theorem and sketch the $Q$-$H$ decomposition
   that proves it; identify which inequality bounds each term.
5. Predict EM's behavior under three initializations: symmetric,
   slightly asymmetric, strongly asymmetric.
6. Identify the role of $k(z \mid x, \theta^{(t)})$ — that it's the
   "soft-imputation" distribution under which the E-step expectation is
   taken.

If a friendly TA quizzed the learner on the above and they failed two or
more, the page didn't do its job.

---

## 9. Stretch Goals (post-MVP)

- **"Build your own data"**: lets the user add/remove trials and re-run EM.
- **EM vs hard-EM comparison**: show that hard-assignment (assigning each
  trial to its argmax coin) gives degenerate behavior on edge cases.
- **Animated derivation**: each algebraic step in §6's M-step derivation
  appears one at a time on a "Next" button click.
- **Audio narration toggle**: TTS reads the prose for accessibility.
- **Dark mode polish**: ensure all viz colors hold up; tune the contour
  fill ramp.
- **Export**: a "save this run" button that downloads the iteration
  history as JSON or CSV.

---

## 10. Out of Scope (intentionally)

- Generalized EM (M-step finds *any* improvement, not the max). Mention
  briefly in §9 only.
- Stochastic EM, MCEM, variational EM. Out of scope; a follow-up lesson.
- The textbook Poisson rate example (Casella & Berger Example 7.2.17–19).
  Replaced by two-coins because two-coins is more visualizable.
- A formal definition of identifiability — alluded to via "label
  switching" but not fully treated.
