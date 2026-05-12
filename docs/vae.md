# Variational Autoencoders (VAE) — Interactive Lesson
## Build Specification & Content Plan

> A deep, interactive lesson on the VAE: the deep-learning embodiment
> of variational inference. By the end, the reader knows how a VAE is
> trained end-to-end, why the loss has the form it does, what the
> latent space looks like, and what failure modes to expect. Concludes
> with a fully interactive trained VAE on a 2D synthetic dataset that
> the reader can probe, sample from, and interpolate within.
>
> **Position in the roadmap**: applications tier (tier 3). The first
> "real generative model" lesson. Required for DDPM. Builds directly
> on ELBO & VI (Form 2 of the ELBO is the VAE loss), the Gaussian
> Cookbook (diagonal-Gaussian KL + reparameterization trick), and
> indirectly on EM (the conceptual ancestor of amortized inference).

---

## 0. Pedagogical Philosophy

Same commitments as the rest of StatViz:

1. **Concrete before abstract.** The lesson opens with the question
   "what would it take to learn a model of digit images?" before any
   VAE formula appears.
2. **Math is shown in full.** The ELBO is re-derived for the VAE
   setting; the amortization step is justified; the encoder/decoder
   roles are traced through a numerical example before the visual
   demo.
3. **The visualization is the climax.** A trained 2D-latent VAE on a
   synthetic dataset, with interactive encoding, sampling,
   interpolation, and decoder probing. This is the lesson where
   probabilistic ML "clicks" for many learners — the viz has to
   deliver.
4. **Cross-page integration is dense.** The VAE lesson is a
   confluence: it picks up the ELBO objective, the Gaussian Cookbook
   identities, and the latent-variable framing from EM. Every major
   piece of machinery is back-linked to where it was developed.

By the end, the reader can: (a) state the VAE objective and identify
each term as either reconstruction or KL regularizer, (b) explain why
the reparameterization trick is necessary and not just convenient,
(c) trace a single training step end-to-end, (d) predict what
happens to a VAE under posterior collapse and how β-VAE addresses it,
(e) interpret a trained VAE's latent space.

---

## 1. Tech Stack

Same Vite multi-page setup as the rest of StatViz. Two new
considerations for this lesson:

- **Tiny neural network runtime.** The §8 visualization runs a
  pre-trained MLP encoder/decoder in the browser. Use vanilla
  TypeScript matrix multiplication (the architecture is small enough
  that no autograd or framework is needed — just forward passes).
  Already-installed `ml-matrix` covers it.
- **Pre-trained weights as static asset.** Ship a JSON blob
  (`/lessons/vae/assets/vae-weights.json`) containing the encoder and
  decoder weights for the §8 model. The training is done offline (see
  §11 below).

File layout:

```
src/lessons/vae/
├── main.ts
├── meta.ts
├── math/
│   ├── elbo.ts                      # the VAE-specific ELBO computation
│   ├── encoder-decoder.ts           # forward passes through a small MLP
│   ├── single-step-trace.ts         # symbolic trace of one training step
│   └── *.test.ts
├── sections/
│   ├── 01-hook.ts
│   ├── 02-the-setup.ts
│   ├── 03-amortized-inference.ts
│   ├── 04-the-vae-objective.ts
│   ├── 05-closed-form-kl.ts
│   ├── 06-reparameterization.ts
│   ├── 07-one-training-step.ts
│   ├── 08-trained-vae-explorer.ts   # the centerpiece visualization wraps this
│   ├── 09-failure-modes.ts
│   └── 10-where-youll-see-this.ts
├── viz/
│   ├── encoder-decoder-diagram.ts   # §2 — architectural illustration
│   ├── amortization-cost.ts         # §3 — per-example optimization vs amortized
│   ├── loss-decomposition.ts        # §4 — reconstruction + KL pie / bar
│   ├── training-trace.ts            # §7 — one training step animated
│   ├── trained-vae-explorer.ts      # §8 — THE centerpiece
│   └── beta-vae-slider.ts           # §9 — β-VAE behavior
├── assets/
│   └── vae-weights.json             # pre-trained model
└── styles/
    └── overrides.css
```

---

## 2. Visual / Aesthetic Direction

Same paper-and-ink aesthetic. Lesson-local accents:

```css
--data-x:      #b8651a;   /* data x — burnt sienna */
--latent-z:    #2c5f8d;   /* latent z — slate blue */
--encoder:     #5a8a6a;   /* encoder — sage */
--decoder:     #c87f3b;   /* decoder — ochre */
--prior:       #6b3a8c;   /* the standard normal prior */
--recon:       #d4a437;   /* reconstruction term — amber */
--kl-reg:      #b54050;   /* KL regularizer term — clay */
```

Two consistent visual conventions used throughout:
- **Data ($x$) and reconstructions ($\hat x$)** in `--data-x`.
- **Latents ($z$) and prior samples** in `--latent-z`.
- The **encoder** is drawn as a left-to-right transformation; the
  **decoder** as right-to-left (when shown together in §8, this
  forms a "data → latent → data" pipeline).

---

## 3. Lesson Metadata (`src/lessons/vae/meta.ts`)

```ts
import type { LessonMeta } from '@shared/system';

export const meta: LessonMeta = {
  id: 'vae',
  title: 'Variational Autoencoders',
  subtitle: 'Deep generative models, fit by gradient ascent on the ELBO.',
  tier: 3,
  difficulty: 3,
  estimatedHours: 4,
  status: 'planned',
  prerequisites: [
    { id: 'elbo-vi',          strength: 'required',    anchor: 'elbo-two-forms' },
    { id: 'gaussian-cookbook',strength: 'required',    anchor: 'kl-mvn-diag' },
    { id: 'em',               strength: 'recommended', anchor: 'q-function' },
  ],
  recommendedNext: ['score-matching', 'ddpm'],
  alsoUsedBy: ['ddpm'],
  description:
    'The variational autoencoder: an amortized variational-inference ' +
    'model with neural-network encoder and decoder, trained end-to-end ' +
    'by gradient ascent on the ELBO. Establishes the architectural ' +
    'pattern that DDPM extends.',
  exportedAnchors: {
    'vae-objective':       'The VAE objective (ELBO with neural encoder/decoder)',
    'amortization':        'Amortized variational inference',
    'reparam-in-vae':      'The reparameterization trick in the VAE pipeline',
    'training-step':       'One full training step, traced numerically',
    'posterior-collapse':  'Posterior collapse: what goes wrong, why, how β-VAE helps',
    'latent-interpolation':'Latent-space interpolation in a trained VAE',
  },
  path: '/lessons/vae',
};
```

---

## 4. Section-by-Section Plan

Ten sections. Plan for ~90–120 minutes of careful reading + interaction.

---

### Section 1 — Hook

**Length**: ~150 words. Full-width banner with a small SVG showing
the data-encoder-latent-decoder-reconstruction pipeline as a sequence
of four boxes connected by arrows.

**Prose** (verbatim):

> Suppose you have a million images of handwritten digits, and you
> want a probabilistic model that captures the structure of "what
> digit images look like." The model should be able to **generate**
> new plausible digits, **interpolate** between two digits smoothly,
> and represent each digit by a small set of numbers.
>
> The challenge: the data $x$ lives in a 784-dimensional pixel space
> (28 × 28). The structure isn't in any single pixel — it's in
> patterns spanning many pixels at once. The model needs a **latent
> representation** $z$ in some lower-dimensional space that captures
> the underlying structure, plus a way to map between $z$ and $x$.
>
> The **variational autoencoder** is one of the elegant answers. It
> trains an *encoder* that maps $x$ to a distribution over $z$, a
> *decoder* that maps $z$ back to $x$, and a *prior* on $z$ that
> structures the latent space — all jointly, by maximizing a single
> objective: the ELBO.

CTA button: "Set up the model →"

---

### Section 2 — The Setup

**Length**: ~500 words. Establishes the generative model.

**Prose**:

> #### The generative model
>
> A VAE specifies a generative process:
>
> 1. **Sample a latent**: $z \sim p(z)$, where the prior $p(z)$ is
>    typically $\mathcal{N}(0, I_d)$ for some small latent dimension
>    $d$ (e.g., $d = 2$ to $d = 100$).
> 2. **Decode to data**: $x \sim p_\theta(x \mid z)$, where
>    $p_\theta(x \mid z)$ is parameterized by a **decoder neural
>    network** with parameters $\theta$.
>
> The decoder output is typically:
> - Gaussian: $p_\theta(x \mid z) = \mathcal{N}\!\big(\mu_\theta(z), \sigma^2 I\big)$
>   for continuous data — the decoder predicts a mean.
> - Bernoulli: $p_\theta(x \mid z) = \prod_i \mathrm{Bern}\!\big(\mathrm{sigmoid}(\mathrm{logits}_\theta(z))_i\big)$
>   for binary images — the decoder predicts pixel logits.
>
> The model is **completely specified** by the prior $p(z)$ and the
> decoder $p_\theta(x \mid z)$. The encoder is *not* part of the
> generative story — it's a tool we'll introduce in a moment for
> *fitting* the model.

> #### The learning objective
>
> Given data $\{x_1, \ldots, x_N\}$, we want $\theta$ that maximizes
> the log-likelihood:
>
> $$\theta^* \;=\; \arg\max_\theta \, \sum_{i=1}^{N} \log p_\theta(x_i) \;=\; \arg\max_\theta \, \sum_{i=1}^{N} \log \int p_\theta(x_i \mid z) p(z) \, dz$$
>
> Each integral is intractable: $z$ is continuous, $p_\theta(x \mid z)$
> is a neural network, no closed form. This is exactly the
> latent-variable-model intractability we met in
> [ELBO & VI §2](../elbo-vi/#the-setup).

**Cross-link callout — back to ELBO** (`type=back`):

> **Uses: [ELBO & VI §2](../elbo-vi/#the-setup)** — the latent-variable
> setup is the same; we just have a neural network in the decoder.
> The intractability is the same. The remedy will be the same too:
> replace $\log p_\theta(x)$ with the ELBO.

**Visualization 1 — `<EncoderDecoderDiagram>`** (medium width):

A static architectural diagram. Two boxes, four arrows:

```
       Data space (x)              Latent space (z)              Data space (x_hat)
       ┌────────────┐              ┌────────────┐              ┌────────────┐
       │            │   encoder    │            │   decoder    │            │
       │     x      │ ───────────► │ q_φ(z | x) │  sample z    │ p_θ(x | z) │
       │            │              │            │ ───────────► │            │
       └────────────┘              └────────────┘              └────────────┘
                                          ▲
                                          │
                                          │ KL                 
                                    ┌─────┴──────┐
                                    │   prior    │
                                    │   p(z)     │
                                    └────────────┘
```

The encoder, decoder, and prior boxes are colored per the tokens above.
Hovering a box reveals a tooltip with its specification (e.g., the
decoder tooltip: "MLP with parameters θ, outputs the mean of a
Gaussian likelihood"). Hovering an arrow reveals what flows through
it. Static visualization — no interaction needed.

---

### Section 3 — Amortized Inference

**Length**: ~500 words. **Anchor: `amortization`**. The conceptual
leap from VI to VAE.

**Prose**:

> #### The problem with plain VI
>
> In [ELBO & VI §5](../elbo-vi/#vi-algorithm) we maximized
> $\mathrm{ELBO}(q)$ over a tractable family $q$. For **one** data
> point $x$, that's a finite optimization problem — fit $q(z)$ once.
> But a VAE trains on *millions* of data points. Doing a separate
> optimization for each $x$ is hopeless:
>
> - $N$ data points → $N$ separate optimizations of a $q_{\phi_i}(z)$
>   per example.
> - Each optimization requires gradient steps. With $N = 10^6$ and
>   100 steps each, that's $10^8$ inner-loop updates per outer-loop
>   parameter update.
>
> The classical variational lower bound on $\log p(x)$ is built
> per-data-point, but training a generative model needs aggregate
> updates over many examples.

> #### Amortization: pay once, query many
>
> The fix is structurally simple but conceptually big. Instead of
> learning a separate $q_{\phi_i}(z)$ for each $x_i$, learn **one
> neural network** $q_\phi(z \mid x)$ that maps any data point $x$ to
> the parameters of a distribution over $z$:
>
> $$\boxed{\;\; q_\phi(z \mid x) \;=\; \mathcal{N}\!\big(\mu_\phi(x), \;\; \mathrm{diag}(\sigma_\phi^2(x))\big) \;\;}$$
>
> where $\mu_\phi(x) \in \mathbb{R}^d$ and $\log \sigma_\phi(x) \in \mathbb{R}^d$
> are outputs of an **encoder network** with parameters $\phi$. (We
> output $\log \sigma$ rather than $\sigma$ so the variance is always
> positive.)
>
> Now there's one set of parameters $\phi$, shared across all data
> points. The neural network has "amortized" the per-data-point
> inference cost into a one-time training cost.
>
> #### The new objective
>
> Maximize the per-example ELBO **averaged over the data**, jointly in
> $\theta$ (the decoder) and $\phi$ (the encoder):
>
> $$(\theta^*, \phi^*) \;=\; \arg\max_{\theta, \phi} \; \frac{1}{N}\sum_{i=1}^{N} \mathrm{ELBO}(x_i; \theta, \phi)$$
>
> Both networks are trained simultaneously by stochastic gradient
> ascent. The encoder learns to produce useful posterior approximations;
> the decoder learns to produce data that matches the observed
> samples.

**Callout — "Why this is impressive"** (`type=tip`):

> Note what amortization gives you: at training time, you pay once
> per example to update the encoder. At inference time (when you want
> $q(z \mid x)$ for a *new* example), you just call $\mu_\phi(x),
> \sigma_\phi(x)$ — one forward pass, no inner-loop optimization.
> Compared to running 100 gradient steps per query, this is a
> ~100× speedup, traded against the inevitable approximation: a
> single network can't be perfectly accurate for every conceivable
> input. **The amortization gap** — the loss in ELBO quality compared
> to per-example optimization — is the price.

**Visualization 2 — `<AmortizationCost>`** (medium width):

A small comparison animation. Left panel: a series of 10 different
data points, each requiring a separate optimization (a curve climbing
to convergence, restarting at each new point). Right panel: a single
neural network that takes any of the same data points and outputs
$(\mu, \sigma)$ in a single forward pass. The right panel's "cost
clock" is one-time; the left panel's accumulates linearly with the
number of points.

The animation drives home the practical point that amortization is
what makes VI scalable to deep-learning data scales.

---

### Section 4 — The VAE Objective

**Length**: ~500 words. **Anchor: `vae-objective`**.

**Prose**:

> Write out the per-example ELBO using Form 2 from
> [ELBO & VI §4](../elbo-vi/#elbo-two-forms):
>
> $$\mathrm{ELBO}(x; \theta, \phi) \;=\; \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x \mid z)\right]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{regularizer}}$$
>
> Both terms have natural interpretations:
>
> #### Reconstruction term
>
> $$\mathbb{E}_{q_\phi(z \mid x)}\!\big[\log p_\theta(x \mid z)\big]$$
>
> Sample $z$ from the encoder's distribution; ask how well the
> decoder's distribution explains $x$ when conditioned on that $z$.
> High value means: **if you encode $x$ to $z$ and decode back, you
> recover $x$ well**. This is the "autoencoder" half of the VAE name.
>
> For a Gaussian decoder $p_\theta(x \mid z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)$,
> the reconstruction term is (up to a constant):
>
> $$\mathbb{E}_{q_\phi(z \mid x)}\!\big[\log p_\theta(x \mid z)\big] \;=\; -\frac{1}{2\sigma^2} \mathbb{E}_{q_\phi(z \mid x)}\!\big[\|x - \mu_\theta(z)\|^2\big] \;+\; \text{const}$$
>
> So maximizing the reconstruction term is **minimizing expected
> squared error** between $x$ and the decoder's mean.

> #### Regularizer term
>
> $$D_{\mathrm{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big)$$
>
> The reverse KL between the encoder's posterior approximation and
> the prior. High value means $q_\phi$ has wandered far from the
> prior. **Minimizing this term pulls every $q_\phi(z \mid x)$
> toward the standard normal $p(z) = \mathcal{N}(0, I)$**.
>
> Why we want this:
> - At sampling time, we generate $z \sim p(z)$ and decode. For
>   decoded samples to look like real data, $z$ values drawn from the
>   prior need to "match" the $z$ values the encoder produced from real
>   data. The regularizer ensures these populations overlap.
> - It also prevents the encoder from cheating by mapping each $x$ to
>   a delta function at a unique $z$ (which would maximize
>   reconstruction at the cost of structure).

> #### The trade-off
>
> The two terms pull in opposite directions:
>
> - **Reconstruction wants** $q_\phi(z \mid x)$ to be sharp around a
>   $z$ that the decoder maps reliably back to $x$.
> - **Regularizer wants** $q_\phi(z \mid x)$ to be near $\mathcal{N}(0, I)$
>   regardless of $x$ — broad, generic.
>
> The trained model finds a compromise. In a well-trained VAE, the
> encoder is *informative enough* to enable reconstruction but
> *broad enough* that the marginal distribution of $z$-samples across
> data approximates the prior.

**Visualization 3 — `<LossDecomposition>`** (medium width):

A horizontal stacked bar chart. The total $-\mathrm{ELBO}$ (the loss)
is decomposed into reconstruction loss and KL loss. The user adjusts
two sliders: "reconstruction quality" and "KL closeness to prior". As
the user changes the sliders, the bar segments shrink and grow.
Annotations show the consequences:
- High reconstruction + high KL → "overfit" (good reconstruction, bad
  generation).
- Low reconstruction + low KL → "posterior collapse" (good prior
  matching, terrible reconstruction).
- Balanced → "healthy VAE".

This is illustrative, not derived from real model. It builds
intuition for the trade-off before §8 shows it concretely.

---

### Section 5 — The Closed-Form KL Term

**Length**: ~300 words. The mechanical detail that makes the KL term
trivially differentiable.

**Prose**:

> The KL term in the VAE loss has a closed form because both
> distributions are Gaussian. From
> [Gaussian Cookbook §3](../gaussian-cookbook/#kl-mvn-diag):
>
> $$D_{\mathrm{KL}}\!\big(\mathcal{N}(\mu, \mathrm{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I)\big) \;=\; \tfrac{1}{2}\sum_{i=1}^{d}\!\left[\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2\right]$$
>
> Substitute the encoder's outputs $\mu = \mu_\phi(x)$,
> $\log \sigma = \log \sigma_\phi(x)$:
>
> $$\boxed{\;\; D_{\mathrm{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big) \;=\; \tfrac{1}{2} \sum_{i=1}^{d} \!\left[e^{2 \log\sigma_{\phi,i}(x)} \;+\; \mu_{\phi,i}(x)^2 \;-\; 1 \;-\; 2\log\sigma_{\phi,i}(x)\right] \;\;}$$
>
> This is a closed-form expression. Differentiating it with respect to
> $\phi$ (via the chain rule through the encoder network) is standard
> autograd — no Monte Carlo estimate needed for this term. In
> contrast, the reconstruction term *does* need Monte Carlo, because
> the integral is over $z$. The reparameterization trick (next
> section) gives us a differentiable Monte Carlo estimate.

**Worked numerical example**:

> For a 2-dim latent with encoder outputs $\mu = (0.5, -0.2)$ and
> $\log \sigma = (0.1, -0.3)$:
>
> $$D_{\mathrm{KL}} \;=\; \tfrac{1}{2}\!\left[(e^{0.2} + 0.25 - 1 - 0.2) + (e^{-0.6} + 0.04 - 1 + 0.6)\right] \;\approx\; 0.2301$$
>
> Pre-computed; verify in `elbo.test.ts`.

**Cross-link callout — back to Gaussian Cookbook** (`type=back`):

> **Uses: [Gaussian Cookbook §3 diagonal-KL formula](../gaussian-cookbook/#kl-mvn-diag)**
> directly. This formula is *the* identity behind the
> "kl_loss = 0.5 * sum(exp(2*logvar) + mu**2 - 1 - 2*logvar)" line
> in every VAE implementation you'll ever encounter.

---

### Section 6 — The Reparameterization Trick (in the VAE pipeline)

**Length**: ~400 words. **Anchor: `reparam-in-vae`**.

**Prose**:

> The reconstruction term
> $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$ involves an
> expectation over $z \sim q_\phi$. To train via gradient descent, we
> need its gradient with respect to **both** $\theta$ (which is
> straightforward, since $\theta$ only appears inside $\log p_\theta$)
> and $\phi$ (which appears in the *distribution we're sampling
> from*).
>
> The naive thing — sample $z$ from $q_\phi$, compute $\log p_\theta(x \mid z)$,
> backpropagate — fails. The sampling step is not differentiable in
> $\phi$: a different $\phi$ would lead to a different $z$, but the
> sample is just a random number with no derivative.
>
> #### The trick (from Gaussian Cookbook §4)
>
> Rewrite the sample as a deterministic transformation of fixed noise:
>
> $$z \;=\; \mu_\phi(x) \;+\; \sigma_\phi(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$
>
> Now $z$ is a deterministic function of $\phi$ (and $\varepsilon$,
> but $\varepsilon$ is just noise — no gradients). The Monte Carlo
> gradient estimate is:
>
> $$\nabla_\phi \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] \;\approx\; \nabla_\phi \log p_\theta\!\big(x \mid \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon\big)$$
>
> with $\varepsilon$ sampled once per gradient step. The gradient
> flows through $\mu_\phi(x), \sigma_\phi(x)$ — and hence through the
> encoder parameters $\phi$ — via standard backprop.

> #### One-line VAE training (in pseudo-code)
>
> ```
> for each minibatch x:
>     mu, log_sigma  = encoder(x; phi)
>     eps            = randn(latent_dim)            # noise
>     z              = mu + exp(log_sigma) * eps    # reparameterization
>     x_hat          = decoder(z; theta)
>     recon_loss     = 0.5 * ||x - x_hat||^2 / sigma_x^2   (or BCE for Bernoulli)
>     kl_loss        = 0.5 * sum(exp(2*log_sigma) + mu**2 - 1 - 2*log_sigma)
>     loss           = recon_loss + kl_loss         # (negative ELBO, averaged)
>     loss.backward()
>     optimizer.step()
> ```
>
> That's all. Eight lines, end-to-end differentiable.

**Cross-link callout — back to Gaussian Cookbook** (`type=back`):

> **Uses: [Gaussian Cookbook §4](../gaussian-cookbook/#reparam-matrix)** —
> the matrix-form reparameterization trick, specialized to diagonal
> covariance, is what makes the line `z = mu + exp(log_sigma) * eps`
> differentiable.

---

### Section 7 — One Training Step, Traced

**Length**: ~600 words. **Anchor: `training-step`**.

**Prose**:

> Let's trace a single training step end-to-end with concrete numbers.
> The full pipeline:
>
> 1. Forward through encoder
> 2. Reparameterize to sample $z$
> 3. Forward through decoder
> 4. Compute reconstruction loss
> 5. Compute KL loss
> 6. Sum to get total loss; backpropagate.

#### Setup
>
> Use a tiny architecture:
> - Data dim: 2 ($x \in \mathbb{R}^2$).
> - Latent dim: 2 ($z \in \mathbb{R}^2$).
> - Encoder: $x \to \tanh(W_e x + b_e) \to (\mu, \log\sigma)$ with
>   hidden dim 4.
> - Decoder: $z \to \tanh(W_d z + b_d) \to \mu_\theta(z)$ with hidden
>   dim 4.
> - Decoder noise: $\sigma_x = 0.1$.

> Take a single data point $x = (1.2, -0.8)$ and one noise sample
> $\varepsilon = (0.5, -0.3)$. The encoder weights are pre-initialized
> (small random values, listed in the worked-example callout).
> We'll compute everything.

#### Step-by-step (rendered as a sequence of styled boxes, each
showing inputs, computation, and output)

> **Step 1: Encoder forward pass.**
>
> $$h_e = \tanh\!\left(W_e x + b_e\right) \in \mathbb{R}^4$$
>
> $$\big(\mu_\phi(x), \log\sigma_\phi(x)\big) = W_o h_e + b_o, \;\; \mu, \log\sigma \in \mathbb{R}^2$$
>
> For the specific (pre-given) weights, this produces
> $\mu \approx (0.34, -0.12)$ and $\log\sigma \approx (-0.2, -0.4)$.

> **Step 2: Reparameterize.**
>
> $$z = \mu + e^{\log\sigma} \odot \varepsilon = (0.34, -0.12) + (0.819, 0.670) \odot (0.5, -0.3) = (0.749, -0.321)$$

> **Step 3: Decoder forward pass.**
>
> $$h_d = \tanh\!\left(W_d' z + b_d'\right) \in \mathbb{R}^4$$
>
> $$\mu_\theta(z) = W_d h_d + b_d$$
>
> For the specific weights, $\mu_\theta(z) \approx (0.95, -0.71)$ —
> a reconstruction.

> **Step 4: Reconstruction loss.**
>
> $$L_{\text{recon}} = \frac{1}{2\sigma_x^2} \|x - \mu_\theta(z)\|^2 = \frac{1}{0.02} \big((1.2 - 0.95)^2 + (-0.8 + 0.71)^2\big) = \frac{1}{0.02}(0.0625 + 0.0081) = 3.53$$

> **Step 5: KL loss.**
>
> $$L_{\text{KL}} = \tfrac{1}{2}\sum_i\!\left[e^{2 \log\sigma_i} + \mu_i^2 - 1 - 2\log\sigma_i\right] = \tfrac{1}{2}\big[(0.670 + 0.116 - 1 + 0.4) + (0.449 + 0.014 - 1 + 0.8)\big] = 0.224$$

> **Step 6: Total loss and backprop.**
>
> $$L_{\text{total}} = L_{\text{recon}} + L_{\text{KL}} = 3.75$$
>
> Differentiate $L_{\text{total}}$ w.r.t. all parameters via the
> standard chain rule. Update parameters by SGD or Adam. The
> reparameterization trick is what makes $\partial L / \partial \phi$
> computable — without it, the gradient would die at Step 2 (the
> sampling step).

**Visualization 4 — `<TrainingTrace>`** (full width):

An animated walkthrough of the six steps above. The user clicks
"Step 1", "Step 2", etc. (or a "Play all" button) and watches:
- Numerical values appear in highlighted boxes at each step.
- A flow diagram on the side highlights which arrow is being
  traversed.
- After Step 6, a small "Backprop" animation shows gradients flowing
  backward through the graph (color-fade the arrows from output back
  to input).

A "Re-randomize noise" button generates a new $\varepsilon$ — the
intermediate $z$ and reconstruction change but the structure stays.
A "Re-randomize weights" button does the same for the weights —
useful for showing that the same procedure works for any parameters.

This visualization is the "math you can trust" demonstration. The §8
viz is the "model you can play with."

---

### Section 8 — A Trained VAE You Can Explore

**Length**: ~700 words. **Anchor: `latent-interpolation`**. The
centerpiece of the lesson.

**Prose**:

> Now let's look at a real (but tiny) VAE trained on a 2D synthetic
> dataset. The architecture is the same as §7: 2D data, 2D latent,
> small MLP encoder/decoder. We trained it for 5000 epochs on 1000
> samples from a four-cluster dataset, with $\beta = 1$.
>
> The trained weights live in `vae-weights.json`. The visualization
> below runs the forward passes (encoder and decoder) in your
> browser — no inference is happening server-side.
>
> #### What to look for
>
> 1. **Encoding**: hover a data point on the data plot. Watch its
>    encoded distribution appear as a small ellipse in the latent
>    plot.
> 2. **Latent structure**: each cluster in data space lands in a
>    coherent region of latent space, **and** all four regions cluster
>    near the origin. This is the prior regularizer working — without
>    it, the encoder would spread clusters across $\mathbb{R}^2$
>    however it pleased.
> 3. **Decoding**: click anywhere in latent space. Watch the decoder
>    produce a reconstruction that appears in data space. Near the
>    origin (high-prior-density region), you get plausible data;
>    far from the origin, the decoder extrapolates.
> 4. **Sampling**: click "Sample from prior" to draw $z \sim \mathcal{N}(0, I)$
>    and decode. Repeated clicks produce diverse plausible data.
> 5. **Interpolation**: click two points in latent space. The viz
>    draws a line between them and shows the decoded data along the
>    line. **Smooth interpolation** in latent space gives smooth
>    morphing in data space — the model has learned a continuous
>    representation.

**Visualization 5 — `<TrainedVAEExplorer>`** (full width, ≥720px tall).
The centerpiece. Layout:

```
┌────────────────────────────────┬────────────────────────────────────┐
│  DATA SPACE (x ∈ R²)            │  LATENT SPACE (z ∈ R²)              │
│                                 │                                    │
│  [scatter of 1000 training      │  [scatter of encoder means         │
│   data points colored by        │   for the same 1000 points,        │
│   true cluster id]              │   colored by cluster id]           │
│                                 │                                    │
│  Hover: highlights point        │  Hover: highlights point           │
│  Click: marks the point         │  Click: marks the point            │
│                                 │                                    │
│  Currently displaying:          │  Currently displaying:             │
│  • the data point you hovered   │  • encoder's μ ± σ ellipse for     │
│  • decoder output for clicked   │    the hovered data point          │
│    latent points                │  • clicked points for decoding     │
└────────────────────────────────┴────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────┐
│  CONTROLS                                                            │
│  [Sample from prior]  [Reconstruct selected]  [Interpolate clicked]  │
│  Beta slider (live):  ◉━━━━━━━━━ 1.0   (see §9 for what this does)  │
└────────────────────────────────────────────────────────────────────┘
```

Behavior details:

- **Sample from prior**: draws 10 $z \sim \mathcal{N}(0, I)$, decodes
  each, plots in data space in `--latent-z`. The decoded points
  should cluster in similar regions as the training data — that's
  the model "generating" new samples.

- **Reconstruct selected**: take the data points the user has clicked,
  encode them, sample $z$ via reparameterization, decode, and overlay
  the reconstructions in `--decoder` color on the data plot.

- **Interpolate clicked**: when the user has clicked exactly two
  latent points, draw a straight line between them, sample 10 evenly
  spaced points along that line, decode each, plot the resulting
  data-space trajectory. Animate the line being traversed.

- **Beta slider**: requires having multiple pre-trained models at
  different $\beta$ values (e.g., $\beta = 0.5, 1, 2, 5$). When the
  user slides, swap in the model for the closest $\beta$. The
  visualization re-renders. **Higher $\beta$ → tighter clustering near
  the origin (more regularization) but worse reconstruction;
  lower $\beta$ → spread-out latent space but better reconstruction.**

#### Pre-training the model (out-of-band)

Training is done offline in Python. The notebook/script that
produces `vae-weights.json` should:

1. Generate a dataset: 1000 points from a 4-cluster Gaussian mixture
   in $\mathbb{R}^2$ (means at $\pm(2, 2)$ and $\pm(2, -2)$, each
   with $\Sigma = 0.2 I$).
2. Define encoder: 2 → 16 → 2(μ) + 2(logσ) with tanh.
3. Define decoder: 2 → 16 → 2 with tanh.
4. Train with Adam (lr = 1e-3) on the VAE loss for 5000 epochs,
   batch size 64.
5. Repeat at $\beta \in \{0.25, 0.5, 1, 2, 5, 10\}$.
6. Save weights as flat JSON arrays for each model.

The training notebook is checked in as
`docs/vae-training-notebook.ipynb` so future updates can reproduce
the model. **The agent building this lesson runs the notebook,
inspects the output, and ships the resulting weights.**

---

### Section 9 — Failure Modes

**Length**: ~500 words. **Anchor: `posterior-collapse`**.

**Prose**:

> Several things can go wrong with a VAE. Two are pedagogically
> important.
>
> #### Posterior collapse
>
> A pathology where the encoder's distribution collapses to the prior:
>
> $$q_\phi(z \mid x) \;\approx\; p(z) \;=\; \mathcal{N}(0, I) \quad \text{for every } x$$
>
> Then **$z$ contains no information about $x$**. The decoder ignores
> $z$ and just outputs the mean of the data distribution. The KL term
> is at its minimum (zero) and the reconstruction term has bottomed
> out at whatever a non-informative $z$ can achieve.
>
> **Why this happens**: the optimization can find a "lazy" minimum
> where the encoder gives up on encoding anything and the decoder
> learns a constant. Especially likely when the decoder is very
> expressive — it can produce useful reconstructions without any help
> from $z$.
>
> **Symptoms**:
> - KL loss near zero throughout training.
> - Reconstructions are blurry / generic, regardless of input.
> - Sampling from the prior produces "average" outputs only.

> #### Reconstruction collapse (the opposite pathology)
>
> The encoder overfits: each $x$ maps to a near-delta $q_\phi(z \mid x)$,
> the decoder reconstructs perfectly, but the latent space is so
> fragmented that prior samples decode to nothing meaningful.
>
> **Why this happens**: when the KL regularizer is too weak (e.g.,
> $\beta$ too small), the model trades regularization for
> reconstruction quality.
>
> **Symptoms**:
> - KL loss is high (encoder distributions don't match prior).
> - Reconstructions are excellent.
> - Prior samples generate garbage.

> #### β-VAE: tuning the trade-off
>
> A common fix: scale the KL term by a hyperparameter $\beta$:
>
> $$\mathrm{ELBO}_\beta(x) \;=\; \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] \;-\; \beta \, D_{\mathrm{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big)$$
>
> - $\beta < 1$: weaker regularization → better reconstruction, worse
>   sampling. Drift toward reconstruction collapse.
> - $\beta = 1$: the standard ELBO. The "principled" choice.
> - $\beta > 1$: stronger regularization → "disentangled"
>   representations (Higgins et al. 2017), but worse reconstruction.
>   Drift toward posterior collapse.
>
> $\beta$ is no longer the ELBO of any model — but it's a useful knob.

**Visualization 6 — `<BetaVAESlider>`** (medium width):

A focused view tied to the §8 explorer. Two panels:
- Left: KL loss and reconstruction loss over training (for the
  trained models at different $\beta$). Show how they trade off.
- Right: a "latent space at this $\beta$" snapshot.

As the user moves the $\beta$ slider:
- Low $\beta$: latent space is spread out (poorly matched to prior),
  reconstructions are sharp.
- High $\beta$: latent space is tightly clustered around origin (well
  matched to prior), reconstructions are blurry.
- $\beta = 1$: balanced.

The visualization reinforces §8's centerpiece by isolating the
$\beta$ effect.

---

### Section 10 — Where You'll See This

**Length**: ~250 words.

**Prose**:

> The VAE is the gateway architecture for modern probabilistic
> generative modeling. Direct extensions and applications:
>
> #### Coming next
>
> - **DDPM**. A hierarchical VAE with **$T$ layers** of latents
>   $z_1, z_2, \ldots, z_T$ structured as a Markov chain. The encoder
>   is *fixed* (a known Gaussian noising process); only the decoder
>   is learned. The training objective is still a sum of ELBO terms,
>   each of which is a KL between Gaussians from the
>   [Gaussian Cookbook](../gaussian-cookbook/).
>
> - **Score Matching**. An entirely different generative approach —
>   no encoder at all, no latents. Learns the gradient of the
>   log-density. Combines with VAE-like architectures in modern
>   diffusion models.
>
> #### Adjacent / sidebar
>
> - **VQ-VAE** — discrete latents instead of continuous; lookup
>   instead of sampling.
> - **VAE-GAN hybrids** — use adversarial loss in addition to ELBO
>   for sharper reconstructions.
> - **Hierarchical / ladder VAEs** — multiple latent layers, each
>   capturing a different scale of structure.
> - **β-VAE and friends** — disentanglement literature.

**Visualization 7 — `<RoadmapMini>`** highlighting the lesson and
showing the dependencies converging from KL → ELBO → VAE, plus the
forward path toward DDPM.

---

## 5. Algorithm / Math Implementation

### `src/lessons/vae/math/elbo.ts`

```ts
import { klDiagFromStandard } from '@lessons/gaussian-cookbook/math/kl-mvn';

/** Reconstruction term for a Gaussian decoder with fixed variance. */
export function gaussianReconLogProb(x: number[], x_mean: number[], sigma_x: number): number {
  const d = x.length;
  let s = -0.5 * d * Math.log(2 * Math.PI * sigma_x * sigma_x);
  for (let i = 0; i < d; i++) s += -0.5 * (x[i] - x_mean[i]) ** 2 / (sigma_x * sigma_x);
  return s;
}

/** ELBO for one example given encoder outputs, reparameterized z, and decoder output. */
export function vaeELBO(
  x: number[],
  mu_phi: number[],
  log_sigma_phi: number[],
  mu_theta_z: number[],
  sigma_x: number,
): { recon: number; kl: number; elbo: number } {
  const recon = gaussianReconLogProb(x, mu_theta_z, sigma_x);
  const kl = klDiagFromStandard(mu_phi, log_sigma_phi);
  return { recon, kl, elbo: recon - kl };
}
```

### `src/lessons/vae/math/encoder-decoder.ts`

```ts
/** Forward pass through an MLP with one hidden layer using tanh. */
export interface MLPWeights {
  W1: number[][];  // hidden_dim x input_dim
  b1: number[];    // hidden_dim
  W2: number[][];  // output_dim x hidden_dim
  b2: number[];    // output_dim
}

export function mlpForward(input: number[], weights: MLPWeights): number[] {
  const h = weights.W1.map((row, i) =>
    Math.tanh(row.reduce((s, w, j) => s + w * input[j], 0) + weights.b1[i])
  );
  return weights.W2.map((row, i) =>
    row.reduce((s, w, j) => s + w * h[j], 0) + weights.b2[i]
  );
}

/** VAE encoder: outputs (mu, log_sigma). */
export interface EncoderWeights extends MLPWeights {}
export function encoderForward(x: number[], w: EncoderWeights, latent_dim: number)
  : { mu: number[]; log_sigma: number[] } {
  const out = mlpForward(x, w);
  return { mu: out.slice(0, latent_dim), log_sigma: out.slice(latent_dim) };
}

/** VAE decoder: outputs mu_theta(z). */
export function decoderForward(z: number[], w: MLPWeights): number[] {
  return mlpForward(z, w);
}

/** Reparameterize: z = mu + exp(log_sigma) * eps. */
export function reparameterize(mu: number[], log_sigma: number[], eps: number[]): number[] {
  return mu.map((m, i) => m + Math.exp(log_sigma[i]) * eps[i]);
}
```

### Test cases

- `klDiagFromStandard([0.5, -0.2], [0.1, -0.3])` ≈ 0.2301 (sanity, ported)
- Encoder forward pass on synthetic weights produces deterministic output.
- `vaeELBO(...)` with the §7 numerical example produces:
  - `recon` ≈ −3.53
  - `kl` ≈ 0.224
  - `elbo` ≈ −3.75
- Loading the pre-trained weights JSON, encoding a known training
  point, and decoding back produces a reconstruction within
  reasonable bounds of the input (e.g., $\|x - \hat x\|_2 < 0.5$
  for in-distribution points at $\beta = 1$).

---

## 6. Component Catalog

### Shared (already exist)
Standard chrome.

### Lesson-local
- `<EncoderDecoderDiagram>` (§2) — static architectural overview.
- `<AmortizationCost>` (§3) — per-example vs amortized animation.
- `<LossDecomposition>` (§4) — reconstruction vs KL bar.
- `<TrainingTrace>` (§7) — step-by-step forward pass.
- `<TrainedVAEExplorer>` (§8) — **the centerpiece**.
- `<BetaVAESlider>` (§9) — β trade-off animation.

---

## 7. Page-Level UX

Same as other lessons. `<PrereqStrip>` shows three prereqs: ELBO/VI
(required), Gaussian Cookbook (required), EM (recommended). Two
notes specific to this lesson:

1. **The §8 explorer is the centerpiece** — invest most polish here.
   Real-time encoder/decoder forward passes must feel instant
   (<10ms per click).

2. **The pre-trained weights** for multiple $\beta$ values must load
   eagerly on page entry (one fetch of ~50KB total). The β slider
   should not require any further network requests.

---

## 8. Acceptance Criteria

A learner who has worked through this page should be able to, on a
blank sheet:

1. Write the VAE objective for one example, identifying the
   reconstruction term and the KL regularizer.
2. Explain in two sentences why the reparameterization trick is
   needed (not just convenient).
3. Trace through a single training step: encoder → reparameterize →
   decoder → loss → backprop, with shapes annotated.
4. Identify posterior collapse from its symptoms and explain why
   β > 1 makes it more likely.
5. Predict (qualitatively) the effect of doubling β on a trained VAE.
6. Explain why amortized inference produces an "amortization gap"
   and what determines its size.
7. Given a trained encoder/decoder and a data point, compute
   everything by hand on a calculator (with help from a single
   matrix multiply).

---

## 9. Stretch Goals (post-MVP)

- **Live retraining in browser**: a "Train this VAE" button that
  runs ~100 gradient steps in the browser (autograd via finite
  differences or a tiny hand-rolled backprop) so the user can watch
  the latent space organize itself in real time. Heavy lift but
  immensely satisfying.
- **MNIST mini-VAE**: a second trained model on actual digit data,
  with 8-dim latent. Show the famous "morph 3 to 8" interpolation.
  Requires shipping ~200KB of weights — fine for a static asset.
- **Encoder uncertainty visualization**: in §8, optionally show the
  encoder's *covariance* (not just mean) per data point — small
  ellipses in latent space. Brings the variational nature into focus.
- **The "ELBO surgery" panel**: separate the reconstruction and KL
  contributions per data point, so the user can see which examples
  the model finds "hard" (high recon loss) vs "regular" (low KL).

---

## 10. Out of Scope (intentionally)

- **Conditional VAEs**, **VQ-VAEs**, **VAE-GANs** — each is a real
  topic, none is required for DDPM.
- **Importance-weighted ELBO (IWAE)** — mentioned in passing in
  ELBO/VI; not expanded here.
- **The amortization gap as a research topic** — a sentence mention is
  enough; the modern view (Cremer, Li, Duvenaud 2018) belongs in a
  research-focused page.
- **Disentanglement metrics and impossibility results** — the
  β-VAE section flags the topic, doesn't get into Locatello et al.
- **Normalizing flows as alternative q-families** — adjacent and
  worth a sidebar lesson, but not on the DDPM path.

---

## 11. Training Notebook (offline pre-step)

The lesson ships pre-trained weights. The agent must run a Python
training notebook before §8 can render. The notebook
(`docs/vae-training-notebook.ipynb`) should:

```python
# Pseudocode — agent expands and runs this
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json

# 1. Generate 4-cluster 2D dataset
np.random.seed(0)
centers = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2]])
N_per_cluster = 250
X = np.vstack([np.random.normal(c, 0.2, (N_per_cluster, 2)) for c in centers]).astype(np.float32)
labels = np.repeat(np.arange(4), N_per_cluster)

# 2. Define tiny VAE (latent_dim = 2, hidden = 16)
class TinyVAE(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.enc1 = nn.Linear(2, 16)
        self.enc_mu = nn.Linear(16, 2)
        self.enc_logsigma = nn.Linear(16, 2)
        self.dec1 = nn.Linear(2, 16)
        self.dec2 = nn.Linear(16, 2)
        self.beta = beta
    def encode(self, x):
        h = torch.tanh(self.enc1(x))
        return self.enc_mu(h), self.enc_logsigma(h)
    def reparam(self, mu, log_sigma):
        return mu + torch.exp(log_sigma) * torch.randn_like(mu)
    def decode(self, z):
        h = torch.tanh(self.dec1(z))
        return self.dec2(h)
    def forward(self, x):
        mu, ls = self.encode(x)
        z = self.reparam(mu, ls)
        return self.decode(z), mu, ls
    def loss(self, x):
        x_hat, mu, ls = self(x)
        recon = ((x - x_hat) ** 2).sum(dim=1).mean()
        kl = 0.5 * (torch.exp(2*ls) + mu**2 - 1 - 2*ls).sum(dim=1).mean()
        return recon + self.beta * kl, recon.item(), kl.item()

# 3. Train at each beta
betas = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
weights_out = {}
for beta in betas:
    model = TinyVAE(beta=beta)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(X)
    for epoch in range(5000):
        idx = np.random.choice(len(Xt), 64, replace=False)
        loss, _, _ = model.loss(Xt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    weights_out[f'beta_{beta}'] = {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}

# 4. Save
with open('src/lessons/vae/assets/vae-weights.json', 'w') as f:
    json.dump(weights_out, f)
```

The agent runs this notebook, inspects each trained model's quality
(latent clusters should be visible, reconstructions reasonable), and
saves. **If a trained model looks degenerate (posterior collapse at
β = 10 or reconstruction collapse at β = 0.25), that's the *expected
demonstration* of §9, not a bug.**