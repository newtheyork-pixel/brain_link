// "Remember my eyes": a saved map is a bit off in a new session (posture, light). How should a
// short check-in (5 dots) fix it? Each posture block of the recording stands in for a new session:
// 5 of its dots are the check-in, the other 4 are the test.
//   none    - use the saved map as is
//   shift   - add the mean check-in error (what Recenter does, but from 5 dots instead of 1)
//   refit   - keep every saved sample, add the check-in samples weighted heavily, refit
// Usage: node --max-old-space-size=8192 eval/gaze/refresh.mjs
import { createReadStream, readdirSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { makeReference, rawFeatures, makeLidBasis, featureVector, fitCalibration, fitRidge } from '../../public/gazemodel.js';

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
    blink: Math.max(d.bs.eyeBlinkLeft ?? 0, d.bs.eyeBlinkRight ?? 0), lm });
}
const W = meta.win.w, H = meta.win.h, cw = meta.cam.w, ch = meta.cam.h;
let seg = 0; F.forEach((f, i) => { const p = F[i - 1]; if (!p || f.ph !== p.ph || f.tx !== p.tx || f.ty !== p.ty || f.ph === 'pursuit') seg++; f.seg = seg; });
const s0 = {}; for (const f of F) s0[f.seg] ??= f.t;
for (const f of F) f.fix = f.ph !== 'pursuit' && f.tx != null && f.t - s0[f.seg] >= 500 && f.blink < 0.5;
const cal = F.filter((f) => (f.ph === 'fix1' && f.fix) || (f.ph === 'pursuit' && f.tx != null && f.blink < 0.6));
const ref = makeReference(cal.map((f) => f.lm), cw, ch);
for (const f of F) f.raw = rawFeatures(f.lm, cw, ch, ref);
const lb = makeLidBasis(cal.map((f) => f.raw.lids));
for (const f of F) f.f = featureVector(f.raw, lb);
const pursuit = F.filter((f) => f.ph === 'pursuit' && f.tx != null);
const dotAt = (t) => { let i = pursuit.findIndex((p) => p.t >= t); if (i <= 0) i = 1; const a = pursuit[i - 1], b = pursuit[i] ?? a;
  const u = b.t === a.t ? 0 : Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t))); return [a.tx + u * (b.tx - a.tx), a.ty + u * (b.ty - a.ty)]; };
const saved = fitCalibration(F.filter((f) => f.ph === 'fix1' && f.fix).map((f) => ({ f: f.f, x: f.tx, y: f.ty })),
  pursuit.filter((f) => f.blink < 0.6).map((f) => ({ f: f.f, t: f.t })), dotAt);

const tileOf = (x, y) => Math.min(3, Math.max(0, Math.floor(x / (W / 4)))) + 4 * Math.min(1, Math.max(0, Math.floor(y / (H / 2))));
const median = (a) => { const s = [...a].sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
const errOf = (pred, set) => {
  let e = 0, n = 0; const per = {};
  for (const f of set) { const [x, y] = pred(f.f); e += Math.hypot(x - f.tx, y - f.ty); n++; (per[f.seg] ??= []).push([x, y, f]); }
  let hits = 0, looks = 0;
  for (const g of Object.values(per)) { hits += tileOf(median(g.map((p) => p[0])), median(g.map((p) => p[1]))) === tileOf(g[0][2].tx, g[0][2].ty); looks++; }
  return { e: e / n, hits, looks };
};
const tot = {};
const add = (k, r) => { const t = (tot[k] ??= { e: 0, n: 0, hits: 0, looks: 0 }); t.e += r.e; t.n++; t.hits += r.hits; t.looks += r.looks; };
for (const ph of ['lean_left', 'lean_right', 'far', 'near', 'tiles']) {
  const block = F.filter((f) => f.ph === ph && f.fix);
  const segs = [...new Set(block.map((f) => f.seg))];
  const checkSegs = new Set(segs.slice(0, 5)), check = block.filter((f) => checkSegs.has(f.seg)), test = block.filter((f) => !checkSegs.has(f.seg));
  const base = (f) => saved.model.predict(f);
  const r0 = errOf(base, test);
  const dx = check.reduce((s, f) => s + (f.tx - base(f.f)[0]), 0) / check.length, dy = check.reduce((s, f) => s + (f.ty - base(f.f)[1]), 0) / check.length;
  const r1 = errOf((f) => { const [x, y] = base(f); return [x + dx, y + dy]; }, test);
  const line = [`${ph.padEnd(10)} none ${r0.e.toFixed(0).padStart(4)}px ${r0.hits}/${r0.looks}`, `shift ${r1.e.toFixed(0).padStart(4)}px ${r1.hits}/${r1.looks}`];
  add('none', r0); add('shift', r1);
  for (const w of [5, 20, 50]) {
    const X = [], Y = [];
    for (const p of saved.samples) { X.push(p.f); Y.push([p.x, p.y]); }
    for (const f of check) for (let j = 0; j < w; j++) { X.push(f.f); Y.push([f.tx, f.ty]); }
    const m = fitRidge(X, Y, saved.lambda);
    const r = errOf((f) => m.predict(f), test);
    line.push(`refit x${w} ${r.e.toFixed(0).padStart(4)}px ${r.hits}/${r.looks}`); add(`refit x${w}`, r);
  }
  console.log(line.join('  |  '));
}
console.log('\nmean over the 5 "new sessions":');
for (const [k, t] of Object.entries(tot)) console.log(`  ${k.padEnd(10)} ${(t.e / t.n).toFixed(0).padStart(4)} px   tiles ${t.hits}/${t.looks}`);
