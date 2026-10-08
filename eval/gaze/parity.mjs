// Run public/gazemodel.js over a recording exactly as the app will, and score it the same way
// analyze.py does. Usage: node parity.mjs [recording.jsonl]
import { createReadStream, readdirSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { makeReference, rawFeatures, makeLidBasis, featureVector, fitCalibration } from '../../public/gazemodel.js';

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
console.log(`${path.basename(file)}: ${F.length} frames`);

// presentations + fixation windows (same rule as load.py: 500 ms settle, no blinks)
let seg = 0; F.forEach((f, i) => {
  const p = F[i - 1];
  if (!p || f.ph !== p.ph || f.tx !== p.tx || f.ty !== p.ty || f.ph === 'pursuit') seg++;
  f.seg = seg;
});
const segStart = {}; for (const f of F) segStart[f.seg] ??= f.t;
for (const f of F) f.fix = f.ph !== 'pursuit' && f.tx != null && f.t - segStart[f.seg] >= 500 && f.blink < 0.5;

const calFrames = F.filter((f) => (f.ph === 'fix1' && f.fix) || (f.ph === 'pursuit' && f.tx != null && f.blink < 0.6));
const t0 = performance.now();
const ref = makeReference(calFrames.map((f) => f.lm), cw, ch);
for (const f of F) f.raw = rawFeatures(f.lm, cw, ch, ref);
const lidBasis = makeLidBasis(calFrames.map((f) => f.raw.lids));
for (const f of F) f.f = featureVector(f.raw, lidBasis);

const still = F.filter((f) => f.ph === 'fix1' && f.fix).map((f) => ({ f: f.f, x: f.tx, y: f.ty }));
const pursuit = F.filter((f) => f.ph === 'pursuit' && f.tx != null);
const moving = pursuit.filter((f) => f.blink < 0.6).map((f) => ({ f: f.f, t: f.t }));
const dotAt = (t) => {
  let i = pursuit.findIndex((p) => p.t >= t);
  if (i <= 0) i = 1;
  const a = pursuit[i - 1], b = pursuit[i] ?? a;
  const u = b.t === a.t ? 0 : Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t)));
  return [a.tx + u * (b.tx - a.tx), a.ty + u * (b.ty - a.ty)];
};
const cal = fitCalibration(still, moving, dotAt);
console.log(`fit in ${Math.round(performance.now() - t0)} ms: lag ${cal.lag} ms, lambda ${cal.lambda}, still-dot err ${cal.stillErrPx}px`);

// runtime: EMA 0.3 on features, then predict
let sm = null;
for (const f of F) { sm = sm ? sm.map((v, k) => v + 0.3 * (f.f[k] - v)) : [...f.f]; f.p = cal.model.predict(sm); }
const tile = (x, y) => Math.min(3, Math.max(0, Math.floor(x / (W / 4)))) + 4 * Math.min(1, Math.max(0, Math.floor(y / (H / 2))));
const med = (a) => { const s = [...a].sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
for (const set of [['fix2'], ['tiles'], ['lean_left', 'lean_right', 'far', 'near']]) {
  const sel = F.filter((f) => f.fix && set.includes(f.ph));
  const ex = sel.reduce((s, f) => s + Math.abs(f.p[0] - f.tx), 0) / sel.length;
  const ey = sel.reduce((s, f) => s + Math.abs(f.p[1] - f.ty), 0) / sel.length;
  let hits = 0, looks = 0;
  for (const s of new Set(sel.map((f) => f.seg))) {
    const g = sel.filter((f) => f.seg === s);
    hits += tile(med(g.map((f) => f.p[0])), med(g.map((f) => f.p[1]))) === tile(g[0].tx, g[0].ty); looks++;
  }
  console.log(`${set.join('+').padEnd(32)} err ${ex.toFixed(0)},${ey.toFixed(0)}px  tiles ${hits}/${looks}`);
}
