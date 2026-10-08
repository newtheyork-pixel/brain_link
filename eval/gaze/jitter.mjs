// How shaky is the dot, and what steadies it without making it slow? Replays a recording through
// the real model + display pipeline (gazemodel.js) and scores each smoothing setup on:
//   shake  - how far the DISPLAYED dot moves per second while he holds still on a dot (px/s)
//   breaks - fixation-lock breaks per 10 s of holding still (each one un-arms the tile)
//   locked - share of held-still frames the dot is locked
//   reach  - ms from the dot appearing somewhere new until the displayed point is within 150 px
//   tiles  - looks landing on the right tile (median over each look), as in parity.mjs
// Usage: node --max-old-space-size=8192 eval/gaze/jitter.mjs [recording.jsonl]
import { createReadStream, readdirSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { makeReference, rawFeatures, makeLidBasis, featureVector, fitCalibration, makeMedian, makeFixation } from '../../public/gazemodel.js';
import { makeBlinkGate } from '../../public/blinkgate.js';

const dir = path.resolve(import.meta.dirname, '../../data/recordings');
const file = process.argv[2] ?? path.join(dir, readdirSync(dir).filter((f) => f.endsWith('.jsonl'))
  .sort((a, b) => statSync(path.join(dir, a)).mtimeMs - statSync(path.join(dir, b)).mtimeMs).pop());
let meta; const F = [];
for await (const line of createInterface({ input: createReadStream(file) })) {
  const d = JSON.parse(line);
  if (d.k === 'meta') meta = d;
  if (d.k !== 'f') continue;
  const lm = [];
  for (let i = 0; i < 478; i++) lm.push({ x: d.lm[i * 3] / 1e5, y: d.lm[i * 3 + 1] / 1e5, z: d.lm[i * 3 + 2] / 1e5 });
  F.push({ t: d.t, ph: d.ph, tx: d.tx == null ? null : d.tx * meta.win.w, ty: d.ty == null ? null : d.ty * meta.win.h,
    L: d.bs.eyeBlinkLeft ?? 0, R: d.bs.eyeBlinkRight ?? 0, lm });
}
const W = meta.win.w, H = meta.win.h, cw = meta.cam.w, ch = meta.cam.h;
let seg = 0; F.forEach((f, i) => { const p = F[i - 1]; if (!p || f.ph !== p.ph || f.tx !== p.tx || f.ty !== p.ty || f.ph === 'pursuit') seg++; f.seg = seg; });
const segStart = {}; for (const f of F) segStart[f.seg] ??= f.t;
for (const f of F) { f.age = f.t - segStart[f.seg]; f.fix = f.ph !== 'pursuit' && f.tx != null && f.age >= 600 && Math.max(f.L, f.R) < 0.5; }

// Fit exactly as the app does (fix1 still dots + pursuit).
const cal = F.filter((f) => (f.ph === 'fix1' && f.fix) || (f.ph === 'pursuit' && f.tx != null && Math.max(f.L, f.R) < 0.6));
const ref = makeReference(cal.map((f) => f.lm), cw, ch);
for (const f of F) f.raw = rawFeatures(f.lm, cw, ch, ref);
const lb = makeLidBasis(cal.map((f) => f.raw.lids));
for (const f of F) f.f = featureVector(f.raw, lb);
const pursuit = F.filter((f) => f.ph === 'pursuit' && f.tx != null);
const dotAt = (t) => { let i = pursuit.findIndex((p) => p.t >= t); if (i <= 0) i = 1; const a = pursuit[i - 1], b = pursuit[i] ?? a;
  const u = b.t === a.t ? 0 : Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t))); return [a.tx + u * (b.tx - a.tx), a.ty + u * (b.ty - a.ty)]; };
const fitted = fitCalibration(F.filter((f) => f.ph === 'fix1' && f.fix).map((f) => ({ f: f.f, x: f.tx, y: f.ty })),
  pursuit.filter((f) => Math.max(f.L, f.R) < 0.6).map((f) => ({ f: f.f, t: f.t })), dotAt);
const model = fitted.model;

// One Euro filter (Casiez 2012): smooths hard when still, follows fast when moving.
function oneEuro({ minCutoff = 1.0, beta = 0.007, dCutoff = 1.0 } = {}) {
  let prev = null, dprev = null, tp = null;
  const alpha = (cut, dt) => { const r = 2 * Math.PI * cut * dt; return r / (r + 1); };
  return (v, t) => {
    if (prev === null) { prev = [...v]; dprev = v.map(() => 0); tp = t; return [...v]; }
    const dt = Math.max(1e-3, (t - tp) / 1000); tp = t;
    const out = v.map((x, k) => {
      const dx = (x - prev[k]) / dt;
      dprev[k] += alpha(dCutoff, dt) * (dx - dprev[k]);
      const cut = minCutoff + beta * Math.abs(dprev[k]);
      return prev[k] + alpha(cut, dt) * (x - prev[k]);
    });
    prev = out; return [...out];
  };
}

// A display pipeline = feature smoother -> model -> point smoother -> median -> fixation lock.
function run({ feat, point, medianN = 7, fix = {} }) {
  const gate = makeBlinkGate();
  const fs = feat(), ps = point ? point() : null;
  const med = makeMedian(medianN), fx = makeFixation(fix);
  const out = new Array(F.length).fill(null);
  let last = null;
  for (let i = 0; i < F.length; i++) {
    const f = F[i];
    const g = gate.step(f.t, f.L, f.R);
    if (g.gated) { out[i] = last; continue; }
    const sf = fs(f.f, f.t);
    let [x, y] = model.predict(sf);
    if (ps) [x, y] = ps([x, y], f.t);
    x = Math.max(0, Math.min(W, x)); y = Math.max(0, Math.min(H, y));
    const [mx, my] = med(x, y);
    const [dx, dy, locked] = fx(mx, my, f.t);
    out[i] = last = { x: dx, y: dy, locked };
  }
  return out;
}
const tileOf = (x, y) => Math.min(3, Math.max(0, Math.floor(x / (W / 4)))) + 4 * Math.min(1, Math.max(0, Math.floor(y / (H / 2))));
const med = (a) => { const s = [...a].sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
function score(out) {
  const held = F.map((f, i) => [f, out[i]]).filter(([f, o]) => f.fix && o && f.ph !== 'fix1');
  let travel = 0, dur = 0, breaks = 0, locked = 0;
  for (let k = 1; k < held.length; k++) {
    const [fa, a] = held[k - 1], [fb, b] = held[k];
    if (fa.seg !== fb.seg || fb.t - fa.t > 100) continue;
    travel += Math.hypot(b.x - a.x, b.y - a.y); dur += fb.t - fa.t;
    if (a.locked && !b.locked) breaks++;
  }
  for (const [, o] of held) if (o.locked) locked++;
  // reach: still-dot presentations outside calibration
  const reach = [];
  for (const s of new Set(F.filter((f) => ['fix2', 'tiles', 'lean_left', 'lean_right', 'far', 'near'].includes(f.ph) && f.tx != null).map((f) => f.seg))) {
    const idx = F.map((f, i) => i).filter((i) => F[i].seg === s);
    const hit = idx.find((i) => out[i] && Math.hypot(out[i].x - F[i].tx, out[i].y - F[i].ty) < 150);
    if (hit !== undefined) reach.push(F[hit].t - F[idx[0]].t);
  }
  let hits = 0, looks = 0;
  for (const s of new Set(F.filter((f) => f.fix && f.ph === 'tiles').map((f) => f.seg))) {
    const g = F.map((f, i) => i).filter((i) => F[i].seg === s && F[i].fix && out[i]);
    hits += tileOf(med(g.map((i) => out[i].x)), med(g.map((i) => out[i].y))) === tileOf(F[g[0]].tx, F[g[0]].ty); looks++;
  }
  return { shake: travel / (dur / 1000), breaks: breaks / (dur / 10000), locked: locked / held.length, reach: med(reach), reached: reach.length, hits, looks };
}
const ema = (a) => () => { let s = null; return (v) => (s = s ? s.map((x, k) => x + a * (v[k] - x)) : [...v]); };
const euro = (o) => () => oneEuro(o);
const SETUPS = {
  'today (EMA 0.30, median 7, lock 55/32)': { feat: ema(0.30) },
  'lock 90px, sustained 0ms': { feat: ema(0.30), fix: { moveThresh: 90, holdThresh: 49, breakMs: 0 } },
  'lock 90px, sustained 60ms': { feat: ema(0.30), fix: { moveThresh: 90, holdThresh: 49, breakMs: 60 } },
  'lock 90px, sustained 100ms': { feat: ema(0.30), fix: { moveThresh: 90, holdThresh: 49, breakMs: 100 } },
  'lock 90px, sustained 150ms': { feat: ema(0.30), fix: { moveThresh: 90, holdThresh: 49, breakMs: 150 } },
  'lock 130px, sustained 0ms': { feat: ema(0.30), fix: { moveThresh: 130, holdThresh: 71, breakMs: 0 } },
  'lock 130px, sustained 60ms': { feat: ema(0.30), fix: { moveThresh: 130, holdThresh: 71, breakMs: 60 } },
  'lock 130px, sustained 100ms': { feat: ema(0.30), fix: { moveThresh: 130, holdThresh: 71, breakMs: 100 } },
  'lock 130px, sustained 150ms': { feat: ema(0.30), fix: { moveThresh: 130, holdThresh: 71, breakMs: 150 } },
  'lock 170px, sustained 0ms': { feat: ema(0.30), fix: { moveThresh: 170, holdThresh: 93, breakMs: 0 } },
  'lock 170px, sustained 60ms': { feat: ema(0.30), fix: { moveThresh: 170, holdThresh: 93, breakMs: 60 } },
  'lock 170px, sustained 100ms': { feat: ema(0.30), fix: { moveThresh: 170, holdThresh: 93, breakMs: 100 } },
  'lock 170px, sustained 150ms': { feat: ema(0.30), fix: { moveThresh: 170, holdThresh: 93, breakMs: 150 } },
  'One Euro (0.5, 0.01) + lock 130/71, 100ms': { feat: ema(1), point: euro({ minCutoff: 0.5, beta: 0.01 }), fix: { moveThresh: 130, holdThresh: 71, breakMs: 100 } },
};
console.log(`${path.basename(file)}: window ${W}x${H}, ${F.length} frames, fit delay ${fitted.lag} ms`);
console.log('setup'.padEnd(44), 'shake px/s  breaks/10s  locked  reach ms  tiles');
for (const [name, s] of Object.entries(SETUPS)) {
  const r = score(run(s));
  console.log(name.padEnd(44), r.shake.toFixed(0).padStart(9), r.breaks.toFixed(1).padStart(11), `${(100 * r.locked).toFixed(0)}%`.padStart(7),
    String(r.reach).padStart(9), `  ${r.hits}/${r.looks}`);
}
