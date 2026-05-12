import { renderMath, callout, crosslinkBack } from '@shared/ui';
import { mountAmortizationCost } from '../viz/amortization-cost';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-3';
  sec.className = 'section';
  sec.innerHTML = `
    <div id="amortization"></div>
    <div class="section-label">§3</div>
    <h2>Amortized Inference</h2>
    <div class="prose">
      <h3>The problem with plain VI</h3>
      <p>In <a href="/StatViz/lessons/elbo-vi/#vi-algorithm">ELBO &amp; VI §5</a> we
      maximized $\\mathrm{ELBO}(q)$ over a tractable family $q$. For <em>one</em> data
      point $x$, that is a finite optimization problem — fit $q(z)$ once.
      But a VAE trains on millions of data points. Doing a separate optimization for
      each $x$ is hopeless:</p>
      <ul>
        <li>$N$ data points → $N$ separate optimizations of $q_{\\phi_i}(z)$.</li>
        <li>Each requires hundreds of gradient steps. With $N = 10^6$, that is
        $\\sim 10^8$ inner-loop updates per outer-loop parameter update.</li>
      </ul>

      <h3>Amortization: pay once, query many</h3>
      <p>Instead of learning a separate $q_{\\phi_i}(z)$ for each $x_i$, learn
      <strong>one neural network</strong> $q_\\phi(z \\mid x)$ that maps any data
      point $x$ to the parameters of a distribution over $z$:</p>
      $$\\boxed{\\;\\; q_\\phi(z \\mid x) \\;=\\; \\mathcal{N}\\!\\big(\\mu_\\phi(x),
        \\;\\; \\mathrm{diag}(\\sigma_\\phi^2(x))\\big) \\;\\;}$$
      <p>where $\\mu_\\phi(x) \\in \\mathbb{R}^d$ and
      $\\log \\sigma_\\phi(x) \\in \\mathbb{R}^d$ are outputs of an
      <strong>encoder network</strong> with parameters $\\phi$.
      (We output $\\log \\sigma$ to keep variance positive.)</p>
      <p>Now there is one set of parameters $\\phi$, shared across all data
      points. The encoder has "amortized" the per-data-point inference cost into a
      one-time training cost. The new objective: maximize the per-example ELBO
      averaged over the data, jointly in $\\theta$ (decoder) and $\\phi$ (encoder):</p>
      $$\\boxed{\\;\\; (\\theta^*, \\phi^*) \\;=\\;
        \\arg\\max_{\\theta, \\phi} \\; \\frac{1}{N}\\sum_{i=1}^{N}
        \\mathrm{ELBO}(x_i; \\theta, \\phi) \\;\\;}$$
    </div>

    ${callout('tip', 'The amortization trade-off',
      `<p>At training time you pay once per example to update the encoder.
      At inference time you call $\\mu_\\phi(x), \\sigma_\\phi(x)$ — one forward pass,
      no inner-loop optimization. Compared to 100 gradient steps per query, this is a
      ~100× speedup, at the cost of an <em>amortization gap</em>: a single network
      cannot be perfectly accurate for every conceivable input.</p>`
    )}

    ${crosslinkBack({
      toLesson: 'em',
      toAnchor: 'q-function',
      toAnchorLabel: '5',
      body: `<p>EM's E-step computes the exact posterior $k(z \\mid x, \\theta^{(t)})$
        per example — a fresh optimization at each iteration. Amortization replaces those
        per-example solutions with a shared encoder network. This is conceptually the same
        jump from "solve exactly once" to "approximate once and reuse."</p>`,
    })}

    <div id="viz-amortization-cost" class="viz-medium"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
  mountAmortizationCost(sec.querySelector('#viz-amortization-cost') as HTMLElement);
}
