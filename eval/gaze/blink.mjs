// Replays a recording through the REAL blink gate (public/blinkgate.js) and the real output stage
// (median + fixation lock from public/gazemodel.js), with the app's tile-arming rule on top, and
// checks that look-then-blink can work for him:
//   1. every real blink freezes the gaze before the lid crosses 0.6 (where the old tracker froze),
//   2. a tile armed when a blink starts is the one the blink would say, and is still armed after,
//   3. the gate almost never freezes the gaze when he is just looking (< 1% of settled frames),
//   4. no natural blink is long enough to count as a deliberate "say it" at the default length.
// Run at the camera's full rate and decimated to ~36 and ~24 fps, since his webcam may be slower.
// Usage: node --max-old-space-size=8192 eval/gaze/blink.mjs [recording.jsonl]
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { makeReference, rawFeatures, makeLidBasis, featureVector, fitCalibration, makeMedian, makeFixation }
  from '../../public/gazemodel.js';
import { makeBlinkGate, BLINK_DEFAULTS } from '../../public/blinkgate.js';

const file = process.argv[2]
  ?? path.resolve(import.meta.dirname, '../../data/recordings/2026-10-08T01-37-40-991Z.jsonl');
// The real blinks in that recording, found by eye in its lid trace (both-eye mean lid > 0.3, fast
// rise and fall). The 2.5 s at ~0.38 from 176.8 s is him leaning in, not a blink: it must NOT gate
// for long and must never count as one.
const BLINKS = [43.23, 43.52, 49.46, 49.83, 66.04, 80.42, 92.79, 100.21, 104.50, 151.79, 157.04, 218.32];

let meta; const ALL = [];
for await (const line of createInterface({ input: createReadStream(file) })) {
  const d = JSON.parse(line);
  if (d.k === 'meta') meta = d;
  if (d.k !== 'f') continue;
  const lm = [];
  for (let i = 0; i < 478; i++) lm.push({ x: d.lm[i * 3] / 1e5, y: d.lm[i * 3 + 1] / 1e5, z: d.lm[i * 3 + 2] / 1e5 });
  ALL.push({ t: d.t, ph: d.ph, tx: d.tx == null ? null : d.tx * meta.win.w, ty: d.ty == null ? null : d.ty * meta.win.h,
    L: d.bs.eyeBlinkLeft ?? 0, R: d.bs.eyeBlinkRight ?? 0, lm });
}
const W = meta.win.w, H = meta.win.h, cw = meta.cam.w, ch = meta.cam.h;
for (const f of ALL) f.lid = Math.max(f.L, f.R);
let seg = 0;
ALL.forEach((f, i) => { const p = ALL[i - 1]; if (!p || f.ph !== p.ph || f.tx !== p.tx || f.ty !== p.ty || f.ph === 'pursuit') seg++; f.seg = seg; });
const segStart = {}; for (const f of ALL) segStart[f.seg] ??= f.t;
for (const f of ALL) f.settled = f.ph !== 'pursuit' && f.tx != null && f.t - segStart[f.seg] >= 500;

// The eye -> screen map, fitted exactly as eval/gaze/parity.mjs does.
const calFrames = ALL.filter((f) => (f.ph === 'fix1' && f.settled && f.lid < 0.5) || (f.ph === 'pursuit' && f.tx != null && f.lid < 0.6));
const ref = makeReference(calFrames.map((f) => f.lm), cw, ch);
for (const f of ALL) { f.raw = rawFeatures(f.lm, cw, ch, ref); f.lm = null; }
const lidBasis = makeLidBasis(calFrames.map((f) => f.raw.lids));
for (const f of ALL) f.f = featureVector(f.raw, lidBasis);
const still = ALL.filter((f) => f.ph === 'fix1' && f.settled && f.lid < 0.5).map((f) => ({ f: f.f, x: f.tx, y: f.ty }));
const pursuit = ALL.filter((f) => f.ph === 'pursuit' && f.tx != null);
const dotAt = (t) => {
  let i = pursuit.findIndex((p) => p.t >= t); if (i <= 0) i = 1;
  const a = pursuit[i - 1], b = pursuit[i] ?? a, u = b.t === a.t ? 0 : Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t)));
  return [a.tx + u * (b.tx - a.tx), a.ty + u * (b.ty - a.ty)];
};
const cal = fitCalibration(still, pursuit.filter((f) => f.lid < 0.6).map((f) => ({ f: f.f, t: f.t })), dotAt);
console.log(`${path.basename(file)}: ${ALL.length} frames, fit lag ${cal.lag} ms, still-dot error ${cal.stillErrPx}px`);

// The app's tiles: 4x2 over the window, hit box shrunk to MARGIN (app.js tileUnder).
const tw = W / 4, th = H / 2, MARGIN = 0.72;
const tileUnder = (x, y) => {
  const c = Math.floor(x / tw), r = Math.floor(y / th);
  if (c < 0 || c > 3 || r < 0 || r > 1) return -1;
  return Math.abs(x - (c + 0.5) * tw) <= tw * MARGIN / 2 && Math.abs(y - (r + 0.5) * th) <= th * MARGIN / 2 ? r * 4 + c : -1;
};

// One pass of gaze.js loop() + app.js onGazePoint (blink-confirm). newGate=false is the OLD
// tracker: freeze only while max lid > 0.6, and any unlocked frame un-arms the tile.
function replay(F, newGate) {
  const gate = makeBlinkGate();
  const median = makeMedian(), fixate = makeFixation();
  let lastF = null, lastOut = null, pre = null, snap = null;
  let dwellTile = -1, cand = -1, candN = 0, armed = -1, armedAt = 0;
  const blinks = [];
  for (const f of F) {
    let gated;
    if (newGate) {
      const g = gate.step(f.t, f.L, f.R);
      if (g.entered) {
        pre = lastOut && lastF && f.t - lastOut.t < 200 ? { ...lastOut, f: lastF } : null;
        snap = { tile: armed, at: f.t };
        f.snap = armed;
      }
      for (const e of g.events) if (e.type === 'blink') blinks.push({ t: f.t, ...e, snap: snap?.tile ?? -1, live: armed });
      gated = g.gated;
    } else gated = f.lid > 0.6;
    f.gated = gated; f.armed = armed;
    if (gated) continue;
    if (pre) { lastF = pre.f; median.fill(pre.mx, pre.my); pre = null; }
    lastF = lastF ? lastF.map((v, k) => v + 0.30 * (f.f[k] - v)) : f.f;
    const [px, py] = cal.model.predict(lastF);
    const [mx, my] = median(Math.max(0, Math.min(W, px)), Math.max(0, Math.min(H, py)));
    const [x, y, locked] = fixate(mx, my, f.t);
    lastOut = { x, y, mx, my, locked, t: f.t };
    const i = tileUnder(x, y);
    if (i !== dwellTile) {
      if (i === cand) candN++; else { cand = i; candN = 1; }
      if (candN >= 4) { armed = -1; dwellTile = i; }
    } else {
      cand = i; candN = 0;
      if (i >= 0 && locked) { armed = i; armedAt = f.t; }
      else if (i >= 0 && armed >= 0 && (!newGate || f.t - armedAt > 300)) armed = -1;
    }
    f.armed = armed;
  }
  return blinks;
}

let failed = 0;
const check = (ok, msg) => { console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failed++; };
const ms = (a) => a.map((v) => (typeof v === 'number' ? Math.round(v) : v)).join(' ');

for (const k of [1, 2, 3]) {
  const F = ALL.filter((_, i) => i % k === 0);
  const fps = Math.round(1000 / ((F.at(-1).t - F[0].t) / F.length));
  console.log(`\n=== every ${k} frame(s), ~${fps} fps`);
  const blinks = replay(F, true);

  // 1. The gate freezes the gaze before the lid reaches 0.6 (or before the peak, if it never does).
  const lead = [];
  for (const b of BLINKS) {
    const w = F.filter((f) => f.t >= b * 1000 - 200 && f.t <= b * 1000 + 400);
    const peak = w.reduce((p, f) => (f.lid > p.lid ? f : p));
    const mark = w.find((f) => f.lid > 0.6) ?? peak;
    const g = w.find((f) => f.gated);
    lead.push(g && g.t <= mark.t ? mark.t - g.t : 'MISS');
  }
  check(!lead.includes('MISS'), `gate on before lid > 0.6 (or peak) on all ${BLINKS.length} blinks; lead ms: ${ms(lead)}`);

  // 2. Armed tile at lid-close = the tile the blink says, and still armed 300 ms after release.
  const rows = [];
  for (const b of BLINKS) {
    const i0 = F.findIndex((f) => f.t >= b * 1000 - 200);
    const e = F.slice(i0).findIndex((f) => f.gated) + i0;
    if (e < i0 || !F[e].settled || F[e - 1].armed < 0) continue;
    const tile = F[e - 1].armed;
    let r = e; while (r < F.length && F[r].gated) r++;
    const during = [...new Set(F.slice(e, r).map((f) => f.armed))].join('/');
    const ev = blinks.find((x) => x.t >= F[e].t && x.t <= F[r]?.t);
    const after = F.filter((f) => f.t > F[r]?.t && f.t <= F[r].t + 300);
    const sameTarget = after.every((f) => f.seg === F[e].seg);
    rows.push({ t: b, tile, snap: F[e].snap, during, said: ev ? ev.snap : 'no shut',
      after: sameTarget ? [...new Set(after.map((f) => f.armed))].join('/') : 'target moved' });
  }
  const ok2 = rows.length && rows.every((r) => r.snap === r.tile && r.during === String(r.tile)
    && (r.said === 'no shut' || r.said === r.tile) && (r.after === 'target moved' || r.after === String(r.tile)));
  check(ok2, `armed tile survives every blink on a still target (${rows.length} blinks with a tile armed): `
    + rows.map((r) => `${r.t}s tile ${r.tile}: snapshot ${r.snap}, armed while gated ${r.during}, blink says ${r.said}, 300 ms after ${r.after}`).join('; '));

  // 3. False freezes on settled still-target frames away from any real blink.
  const near = (f) => BLINKS.some((b) => f.t >= b * 1000 - 300 && f.t <= b * 1000 + 900);
  const settled = F.filter((f) => f.settled && !near(f));
  const fg = settled.filter((f) => f.gated).length;
  check(fg / settled.length < 0.01, `false-gated settled frames ${fg}/${settled.length} = ${(100 * fg / settled.length).toFixed(2)}%`);

  // 4. Natural blinks never reach the default confirm length; nothing else counts as a blink.
  const held = blinks.map((x) => `${x.kind}:${Math.round(x.held)}`);
  check(blinks.every((x) => x.kind !== 'long' && x.kind !== 'rest'),
    `no blink reaches ${BLINK_DEFAULTS.confirmMs} ms (natural blinks: ${held.join(' ')})`);
  const stray = blinks.filter((x) => !BLINKS.some((b) => x.t >= b * 1000 - 100 && x.t <= b * 1000 + 900));
  check(!stray.length, `no blink event outside the real blinks${stray.length ? ': ' + stray.map((x) => (x.t / 1000).toFixed(2)).join(' ') : ''}`);

  if (k === 1) {
    // For the record: how often the RIGHT tile is armed while he looks at it, old vs new.
    const share = (G) => {
      const s = G.filter((f) => f.ph === 'tiles' && f.settled);
      return (100 * s.filter((f) => f.armed === tileUnder(f.tx, f.ty)).length / s.length).toFixed(1);
    };
    const Fo = ALL.map((f) => ({ ...f })); replay(Fo, false);
    console.log(`  info: right tile armed on settled 'tiles' frames: old ${share(Fo)}%, new ${share(F)}%`);
  }
}
// The recording has no DELIBERATE closures, so check those on synthetic lids at 30 fps: open at
// `base`, close in 80 ms to (pL, pR), hold, reopen in 150 ms. Small frame noise throughout.
console.log('\n=== synthetic deliberate closures, 30 fps');
let seed = 3;
const noise = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.03;
function closure({ base = 0.15, baseR = base, pL = 0.8, pR = 0.8, hold }) {
  const g = makeBlinkGate(), ev = [];
  const lidAt = (t, b, p) => t < 2000 ? b : t < 2080 ? b + (p - b) * (t - 2000) / 80
    : t < 2080 + hold ? p : t < 2230 + hold ? p - (p - b) * (t - 2080 - hold) / 150 : b;
  for (let t = 0; t < 3500 + hold; t += 33) {
    for (const e of g.step(t, lidAt(t, base, pL) + noise(), lidAt(t, baseR, pR) + noise()).events) ev.push(e);
  }
  return ev;
}
const kinds = (ev) => ev.map((e) => (e.type === 'held' ? 'beep' : `${e.kind}:${Math.round(e.held)}`)).join(' ');
for (const [what, o, want] of [
  ['400 ms hold, both eyes', { hold: 400 }, 'long'],
  ['400 ms hold, weak right eye peaks 0.45 (old rule needed both > 0.6)', { hold: 400, pR: 0.45 }, 'long'],
  ['600 ms hold, droopy lids open at 0.40', { hold: 600, base: 0.4, pL: 0.85, pR: 0.8 }, 'long'],
  ['1100 ms hold', { hold: 1100 }, 'long'],
  ['200 ms natural blink', { hold: 120 }, 'short'],
  ['4 s eyes closed (rest)', { hold: 4000 }, 'rest'],
]) {
  const ev = closure(o), blink = ev.find((e) => e.type === 'blink');
  const beeped = ev.some((e) => e.type === 'held');
  // A rest beeps too: at the beep nobody can know yet that he will keep them shut. Reopening
  // after the rest limit then does nothing.
  const wantBeep = want !== 'short';
  check(blink?.kind === want && beeped === wantBeep, `${what}: ${kinds(ev) || 'nothing'} (want ${want}${wantBeep ? ' + beep' : ''})`);
}
{ // leaning in: both lids creep from 0.10 to 0.45 over 2 s and stay. Never a blink.
  const g = makeBlinkGate(); let ev = 0, gated = 0, n = 0;
  for (let t = 0; t < 6000; t += 33) {
    const v = 0.1 + 0.35 * Math.min(1, Math.max(0, (t - 1000) / 2000)) + noise();
    const r = g.step(t, v, v); ev += r.events.length; n++; if (r.gated) gated++;
  }
  check(ev === 0 && gated / n < 0.1, `slow lean to 0.45: ${ev} blink events, gaze frozen ${(100 * gated / n).toFixed(1)}% of the time`);
}

console.log(failed ? `\n${failed} check(s) FAILED` : '\nPASS');
process.exit(failed ? 1 : 0);
