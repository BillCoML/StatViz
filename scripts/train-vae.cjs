#!/usr/bin/env node
/**
 * VAE training in pure JavaScript (no dependencies).
 * Trains a tiny VAE (2→16→2 latent) on a 2D four-cluster dataset
 * at six beta values, saves weights as JSON for the §8 visualization.
 *
 * Run from project root: node scripts/train-vae.js
 */

'use strict';
const fs   = require('fs');
const path = require('path');

// ─── Seeded PRNG ────────────────────────────────────────────────────────────
let seed = 42;
function rand() {
  seed = (Math.imul(1664525, seed) + 1013904223) | 0;
  return ((seed >>> 0) / 0x100000000);
}
function randn() {
  let u; do { u = rand(); } while (u === 0);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// ─── Data ───────────────────────────────────────────────────────────────────
const CENTERS       = [[2,2],[2,-2],[-2,2],[-2,-2]];
const N_PER_CLUSTER = 250;
const DATA_STD      = 0.2;
const LATENT_DIM    = 2;
const HIDDEN_DIM    = 16;

function generateData(seedVal) {
  seed = seedVal;
  const X = [], labels = [];
  for (let ci = 0; ci < 4; ci++)
    for (let i = 0; i < N_PER_CLUSTER; i++) {
      X.push([CENTERS[ci][0] + DATA_STD * randn(), CENTERS[ci][1] + DATA_STD * randn()]);
      labels.push(ci);
    }
  return { X, labels };
}

// ─── Linear algebra helpers ──────────────────────────────────────────────────
const zeros2d = (r, c) => Array.from({length:r}, () => new Float64Array(c));
const zeros1d = n => new Float64Array(n);

function matvec(W, x) {           // W: m×n, x: n  →  m
  const out = new Float64Array(W.length);
  for (let i = 0; i < W.length; i++)
    for (let j = 0; j < x.length; j++) out[i] += W[i][j] * x[j];
  return out;
}

function vecTW_mul(x, W) {        // x: m  (treated as row), W: m×n  →  n  (W^T x)
  const out = new Float64Array(W[0].length);
  for (let j = 0; j < W[0].length; j++)
    for (let i = 0; i < x.length; i++) out[j] += W[i][j] * x[i];
  return out;
}

function outerAdd(acc, a, b) {    // acc += a ⊗ b  (in place)
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++) acc[i][j] += a[i] * b[j];
}

const tanhV   = v => v.map(Math.tanh);
const dtanhH  = h => h.map(v => 1 - v*v);   // derivative at tanh output h

// ─── Weight initialisation (He-style for tanh) ───────────────────────────────
function initW(rows, cols) {
  const scale = Math.sqrt(2 / (rows + cols));
  const W = Array.from({length: rows}, () => new Float64Array(cols));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) W[i][j] = scale * randn();
  return W;
}

// ─── Model ───────────────────────────────────────────────────────────────────
function makeModel(beta) {
  return {
    beta,
    enc1_W:   initW(HIDDEN_DIM, 2),
    enc1_b:   zeros1d(HIDDEN_DIM),
    encMu_W:  initW(LATENT_DIM, HIDDEN_DIM),
    encMu_b:  zeros1d(LATENT_DIM),
    encLs_W:  initW(LATENT_DIM, HIDDEN_DIM),
    encLs_b:  zeros1d(LATENT_DIM).fill(-1),   // bias toward σ≈0.37 initially
    dec1_W:   initW(HIDDEN_DIM, LATENT_DIM),
    dec1_b:   zeros1d(HIDDEN_DIM),
    dec2_W:   initW(2, HIDDEN_DIM),
    dec2_b:   zeros1d(2),
  };
}

// Adam moments
function makeAdam(model) {
  const keys = Object.keys(model).filter(k => k !== 'beta');
  const adam = {};
  for (const k of keys) {
    const p = model[k];
    if (p instanceof Float64Array) {
      adam[k+'_m'] = zeros1d(p.length);
      adam[k+'_v'] = zeros1d(p.length);
    } else {
      adam[k+'_m'] = p.map(r => new Float64Array(r.length));
      adam[k+'_v'] = p.map(r => new Float64Array(r.length));
    }
  }
  adam.t = 0;
  return adam;
}

// ─── Forward pass ─────────────────────────────────────────────────────────────
function forward(model, x, epsVec) {
  // Encoder
  const pre_e = matvec(model.enc1_W, x).map((v,i) => v + model.enc1_b[i]);
  const h_e   = tanhV(pre_e);
  const mu    = matvec(model.encMu_W, h_e).map((v,i) => v + model.encMu_b[i]);
  const ls    = matvec(model.encLs_W, h_e).map((v,i) => v + model.encLs_b[i]);
  // Reparameterize
  const z     = mu.map((m,i) => m + Math.exp(ls[i]) * epsVec[i]);
  // Decoder
  const pre_d = matvec(model.dec1_W, z).map((v,i) => v + model.dec1_b[i]);
  const h_d   = tanhV(pre_d);
  const x_hat = matvec(model.dec2_W, h_d).map((v,i) => v + model.dec2_b[i]);
  return { h_e, mu, ls, z, h_d, x_hat, epsVec };
}

// ─── Backward pass ────────────────────────────────────────────────────────────
function backward(model, x, cache) {
  const { h_e, mu, ls, z, h_d, x_hat, epsVec } = cache;
  const beta = model.beta;

  // dL/dx_hat (MSE reconstruction)
  const d_xhat = x_hat.map((v,i) => 2*(v - x[i]));

  // Decoder
  const d_dec2_b = d_xhat.slice();
  const d_dec2_W = zeros2d(2, HIDDEN_DIM);
  outerAdd(d_dec2_W, d_xhat, h_d);
  const d_hd     = vecTW_mul(d_xhat, model.dec2_W);        // dec2_W^T @ d_xhat
  const d_pred   = d_hd.map((v,i) => v * (1 - h_d[i]*h_d[i]));  // through tanh

  const d_dec1_b = d_pred.slice();
  const d_dec1_W = zeros2d(HIDDEN_DIM, LATENT_DIM);
  outerAdd(d_dec1_W, d_pred, z);
  const d_z      = vecTW_mul(d_pred, model.dec1_W);         // dec1_W^T @ d_pred

  // KL gradient contributions
  const d_mu = d_z.map((v,i) => v + beta * mu[i]);
  const d_ls = d_z.map((v,i) =>
    v * Math.exp(ls[i]) * epsVec[i] + beta * (Math.exp(2*ls[i]) - 1)
  );

  // Encoder
  const d_encMu_b = d_mu.slice();
  const d_encMu_W = zeros2d(LATENT_DIM, HIDDEN_DIM);
  outerAdd(d_encMu_W, d_mu, h_e);

  const d_encLs_b = d_ls.slice();
  const d_encLs_W = zeros2d(LATENT_DIM, HIDDEN_DIM);
  outerAdd(d_encLs_W, d_ls, h_e);

  const d_he = new Float64Array(HIDDEN_DIM);
  for (let j = 0; j < HIDDEN_DIM; j++) {
    for (let i = 0; i < LATENT_DIM; i++) {
      d_he[j] += model.encMu_W[i][j] * d_mu[i] + model.encLs_W[i][j] * d_ls[i];
    }
  }
  const d_pre_e = d_he.map((v,i) => v * (1 - h_e[i]*h_e[i]));

  const d_enc1_b = d_pre_e.slice();
  const d_enc1_W = zeros2d(HIDDEN_DIM, 2);
  outerAdd(d_enc1_W, d_pre_e, x);

  return { d_enc1_W, d_enc1_b, d_encMu_W, d_encMu_b, d_encLs_W, d_encLs_b,
           d_dec1_W, d_dec1_b, d_dec2_W, d_dec2_b };
}

// ─── Adam update ─────────────────────────────────────────────────────────────
function adamStep(model, adam, grads, lr=0.001, b1=0.9, b2=0.999, eps=1e-8) {
  adam.t++;
  const bc1 = 1 - b1**adam.t, bc2 = 1 - b2**adam.t;

  const upd2d = (key, dkey) => {
    const W = model[key], dW = grads[dkey];
    const mW = adam[key+'_m'], vW = adam[key+'_v'];
    for (let i = 0; i < W.length; i++)
      for (let j = 0; j < W[i].length; j++) {
        mW[i][j] = b1*mW[i][j] + (1-b1)*dW[i][j];
        vW[i][j] = b2*vW[i][j] + (1-b2)*dW[i][j]**2;
        W[i][j] -= lr * (mW[i][j]/bc1) / (Math.sqrt(vW[i][j]/bc2)+eps);
      }
  };
  const upd1d = (key, dkey) => {
    const b = model[key], db = grads[dkey];
    const mb = adam[key+'_m'], vb = adam[key+'_v'];
    for (let i = 0; i < b.length; i++) {
      mb[i] = b1*mb[i] + (1-b1)*db[i];
      vb[i] = b2*vb[i] + (1-b2)*db[i]**2;
      b[i] -= lr * (mb[i]/bc1) / (Math.sqrt(vb[i]/bc2)+eps);
    }
  };

  upd2d('enc1_W','d_enc1_W');  upd1d('enc1_b','d_enc1_b');
  upd2d('encMu_W','d_encMu_W'); upd1d('encMu_b','d_encMu_b');
  upd2d('encLs_W','d_encLs_W'); upd1d('encLs_b','d_encLs_b');
  upd2d('dec1_W','d_dec1_W');  upd1d('dec1_b','d_dec1_b');
  upd2d('dec2_W','d_dec2_W');  upd1d('dec2_b','d_dec2_b');
}

// ─── Training loop ────────────────────────────────────────────────────────────
function trainEpoch(model, adam, X) {
  const n = X.length;
  // shuffle indices
  const idx = Array.from({length:n}, (_,i) => i);
  for (let i = n-1; i > 0; i--) {
    const j = (rand() * (i+1)) | 0;
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }

  const BATCH = 64;
  let totalLoss = 0, batches = 0;

  for (let start = 0; start < n; start += BATCH) {
    const end   = Math.min(start + BATCH, n);
    const bSize = end - start;

    // Accumulate gradients
    const acc = {
      d_enc1_W: zeros2d(HIDDEN_DIM, 2),
      d_enc1_b: zeros1d(HIDDEN_DIM),
      d_encMu_W: zeros2d(LATENT_DIM, HIDDEN_DIM),
      d_encMu_b: zeros1d(LATENT_DIM),
      d_encLs_W: zeros2d(LATENT_DIM, HIDDEN_DIM),
      d_encLs_b: zeros1d(LATENT_DIM),
      d_dec1_W: zeros2d(HIDDEN_DIM, LATENT_DIM),
      d_dec1_b: zeros1d(HIDDEN_DIM),
      d_dec2_W: zeros2d(2, HIDDEN_DIM),
      d_dec2_b: zeros1d(2),
    };

    let bLoss = 0;
    for (let k = start; k < end; k++) {
      const xi  = X[idx[k]];
      const eps = [randn(), randn()];
      const cache = forward(model, xi, eps);
      const { x_hat, mu, ls } = cache;

      const recon = xi.reduce((s,xv,i) => s + (xv-x_hat[i])**2, 0);
      const kl    = 0.5 * ls.reduce((s,lv,i) =>
        s + Math.exp(2*lv) + mu[i]**2 - 1 - 2*lv, 0);
      bLoss += (recon + model.beta * kl) / bSize;

      const grads = backward(model, xi, cache);
      // accumulate (divide by batch size)
      for (const gk of Object.keys(acc)) {
        const g = grads[gk], a = acc[gk];
        if (a instanceof Float64Array) {
          for (let i = 0; i < a.length; i++) a[i] += g[i] / bSize;
        } else {
          for (let i = 0; i < a.length; i++)
            for (let j = 0; j < a[i].length; j++) a[i][j] += g[i][j] / bSize;
        }
      }
    }
    adamStep(model, adam, acc);
    totalLoss += bLoss;
    batches++;
  }
  return totalLoss / batches;
}

// ─── Model export ─────────────────────────────────────────────────────────────
function round6(x) { return Math.round(x * 1e6) / 1e6; }
function roundArr(a) {
  if (typeof a[0] === 'number' || a[0] instanceof Float64Array)
    return Array.from(a instanceof Float64Array ? a : a, v =>
      typeof v === 'number' ? round6(v) : Array.from(v, round6));
  return a.map(roundArr);
}

function exportWeights(model) {
  return {
    'enc1.weight':       roundArr(model.enc1_W),
    'enc1.bias':         roundArr(model.enc1_b),
    'enc_mu.weight':     roundArr(model.encMu_W),
    'enc_mu.bias':       roundArr(model.encMu_b),
    'enc_logsigma.weight': roundArr(model.encLs_W),
    'enc_logsigma.bias': roundArr(model.encLs_b),
    'dec1.weight':       roundArr(model.dec1_W),
    'dec1.bias':         roundArr(model.dec1_b),
    'dec2.weight':       roundArr(model.dec2_W),
    'dec2.bias':         roundArr(model.dec2_b),
  };
}

// ─── Main ─────────────────────────────────────────────────────────────────────
const BETAS  = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
const EPOCHS = 4000;

const { X } = generateData(42);
const out = {};

for (const beta of BETAS) {
  seed = 1337 + Math.round(beta * 100);   // different init per beta
  const model = makeModel(beta);
  const adam  = makeAdam(model);

  const t0 = Date.now();
  for (let e = 0; e < EPOCHS; e++) {
    const loss = trainEpoch(model, adam, X);
    if (e === EPOCHS-1 || e % 1000 === 999)
      process.stdout.write(`  beta=${beta.toFixed(2)}, epoch=${e+1}/${EPOCHS}, loss=${loss.toFixed(4)}\n`);
  }
  process.stdout.write(`  [done in ${((Date.now()-t0)/1000).toFixed(1)}s]\n`);

  out[`beta_${beta}`] = exportWeights(model);
}

out.metadata = {
  centers:       CENTERS,
  n_per_cluster: N_PER_CLUSTER,
  latent_dim:    LATENT_DIM,
  hidden_dim:    HIDDEN_DIM,
  epochs:        EPOCHS,
  training:      'Node.js (scripts/train-vae.js)',
};

const outPath = path.resolve(__dirname, '../src/lessons/vae/assets/vae-weights.json');
fs.writeFileSync(outPath, JSON.stringify(out));
const kb = (fs.statSync(outPath).size / 1024).toFixed(1);
console.log(`\nWeights saved → ${outPath}  (${kb} KB)`);
