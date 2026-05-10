import { renderMath, callout, crosslinkBack } from '@shared/ui';

export function mount(container: HTMLElement): void {
  const sec = document.createElement('section');
  sec.id = 'section-3';
  sec.className = 'section';
  sec.innerHTML = `
    <div class="section-label">§3</div>
    <h2>The ELBO Identity</h2>
    <div class="prose" id="fundamental-identity">
      <p>We'll derive the ELBO two ways. Both are short. The second is more
      informative; the first is the historical motivation for the name.</p>

      <h3>Derivation 1 — via Jensen's inequality</h3>
      <p>Start from the marginal likelihood. Insert any distribution $q(z)$
      on the same support as $p(z \\mid x)$ — multiply and divide, since the
      choice is free:</p>
      $$\\log p(x) \\;=\\; \\log \\int p(x, z) \\, dz \\;=\\; \\log \\int q(z) \\, \\frac{p(x, z)}{q(z)} \\, dz \\;=\\;
      \\log \\mathbb{E}_{Z \\sim q}\\!\\left[\\frac{p(x, Z)}{q(Z)}\\right]$$
      <p>Apply Jensen's inequality. Since $\\log$ is concave, Jensen reverses:</p>
      $$\\log \\mathbb{E}_{Z \\sim q}\\!\\left[\\frac{p(x, Z)}{q(Z)}\\right]
      \\;\\geq\\; \\mathbb{E}_{Z \\sim q}\\!\\left[\\log \\frac{p(x, Z)}{q(Z)}\\right]$$
      <p>Define the right-hand side to be the <strong>Evidence Lower Bound</strong>:</p>
      $$\\boxed{\\;\\; \\mathrm{ELBO}(q) \\;:=\\; \\mathbb{E}_{Z \\sim q}\\!\\left[\\log \\frac{p(x, Z)}{q(Z)}\\right]
      \\;=\\; \\mathbb{E}_q[\\log p(x, Z)] - \\mathbb{E}_q[\\log q(Z)] \\;\\;}$$
      <p>What we just established is that for <em>any</em> $q$,</p>
      $$\\log p(x) \\;\\geq\\; \\mathrm{ELBO}(q)$$
      <p>ELBO is a lower bound on the log evidence. Hence the name.</p>

      <h3>Derivation 2 — via the KL identity</h3>
      <p>Start from the KL divergence between $q$ and the true posterior:</p>
      $$D_{\\mathrm{KL}}(q(z) \\,\\|\\, p(z \\mid x)) \\;=\\; \\mathbb{E}_q\\!\\left[\\log \\frac{q(Z)}{p(Z \\mid x)}\\right]
      \\;=\\; \\mathbb{E}_q[\\log q(Z)] - \\mathbb{E}_q[\\log p(Z \\mid x)]$$
      <p>Substitute $p(z \\mid x) = p(x, z) / p(x)$:</p>
      $$\\mathbb{E}_q[\\log p(Z \\mid x)] \\;=\\; \\mathbb{E}_q[\\log p(x, Z)] - \\log p(x)$$
      <p>($\\log p(x)$ is constant in $Z$ so it pulls outside the expectation.)</p>
      <p>Plugging in:</p>
      $$D_{\\mathrm{KL}}(q \\,\\|\\, p_{\\text{post}}) \\;=\\; \\mathbb{E}_q[\\log q(Z)] - \\mathbb{E}_q[\\log p(x, Z)] + \\log p(x)
      \\;=\\; -\\mathrm{ELBO}(q) + \\log p(x)$$
      <p>Rearranging gives <strong>the fundamental identity</strong>:</p>
      $$\\boxed{\\;\\; \\log p(x) \\;=\\; \\mathrm{ELBO}(q) \\;+\\; D_{\\mathrm{KL}}(q(z) \\,\\|\\, p(z \\mid x)) \\;\\;}$$
      <p>Stop and read this twice. It says:</p>
      <ol>
        <li>$\\mathrm{ELBO}(q) \\leq \\log p(x)$ for every $q$ (since KL ≥ 0).</li>
        <li><strong>The gap is exactly $D_{\\mathrm{KL}}(q \\,\\|\\, p_{\\text{posterior}})$.</strong></li>
        <li>Maximizing ELBO over $q$ is equivalent to minimizing reverse KL to the
        posterior — and at the maximum, ELBO = $\\log p(x)$ if and only if
        $q = p_{\\text{posterior}}$ (almost everywhere).</li>
      </ol>
    </div>

    ${callout('tip', 'Why this identity is so useful',
      `<p>Most of the time, we cannot compute $\\log p(x)$ (that was the whole problem).
      But we can compute $\\mathrm{ELBO}(q)$ — it's an expectation under our chosen $q$,
      of quantities we can evaluate ($\\log p(x, z)$ is the joint, which we know, and
      $\\log q(z)$ is trivial since we picked $q$). So the identity gives us a
      <strong>handle</strong>: maximize an evaluable quantity (ELBO) and you've implicitly
      done two things — you've <strong>found</strong> a good $q$ and you've
      <strong>lower-bounded</strong> $\\log p(x)$ by a known quantity.</p>`
    )}

    ${crosslinkBack({
      toLesson: 'kl-jensen',
      toAnchor: 'gibbs-inequality',
      toAnchorLabel: '5',
      body: `<p>Derivation 1 applies <strong>Jensen's inequality</strong> (§3 of KL &amp; Jensen)
        to the concave $\\log$ function. Derivation 2 uses the fact that
        $D_{\\mathrm{KL}} \\geq 0$ — <strong>Gibbs' inequality</strong> (§5) — to conclude
        that the gap is non-negative. The fundamental identity is the structural reason
        variational inference works at all.</p>`,
    })}

    <div class="prose">
      <h3>The visualization</h3>
      <p>The interactive below shows the three quantities in the identity as a function of
      the variational parameter $\\phi_\\mu$ (the mean of $q = \\mathcal{N}(\\phi_\\mu, 1)$
      approximating a fixed posterior $\\mathcal{N}(1, 1)$). Drag the slider and watch ELBO
      + KL gap always sum to $\\log p(x)$.</p>
    </div>
    <div id="viz-elbo-kl-decomp" class="viz-wide"></div>
  `;
  container.appendChild(sec);
  renderMath(sec);
}
