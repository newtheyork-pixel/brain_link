// Deliberate closures of every speed must register; glances down and leaning in must not.
// Run: node eval/gaze/slowclose.mjs
import { makeBlinkGate } from '../../public/blinkgate.js';

let seed = 3;
const noise = () => (((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5) * 0.03;
const ease = (u) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)));

// lid(t): both-eye blink score over time; R scaled to simulate a weaker eye.
function run(lid, { fps = 30, ms = 4000, weak = 1 } = {}) {
  const g = makeBlinkGate();
  const ev = [];
  for (let t = 0; t < ms; t += 1000 / fps) {
    const v = lid(t);
    const o = g.step(t, Math.min(1, v + noise()), Math.min(1, v * weak + noise()));
    for (const e of o.events) ev.push(e.type === 'held' ? 'beep' : `${e.kind}:${Math.round(e.held)}`);
  }
  return ev;
}
// open 0.12 for 1.5 s, close over `closeMs` to `depth`, hold `holdMs`, reopen over 150 ms
const closure = (closeMs, holdMs, depth = 0.8) => (t) => {
  const t0 = 1500, t1 = t0 + closeMs, t2 = t1 + holdMs, t3 = t2 + 150;
  if (t < t0) return 0.12;
  if (t < t1) return 0.12 + (depth - 0.12) * ease((t - t0) / closeMs);
  if (t < t2) return depth;
  if (t < t3) return depth - (depth - 0.12) * ease((t - t2) / 150);
  return 0.12;
};
const ramp = (to, overMs) => (t) => (t < 1500 ? 0.12 : 0.12 + (to - 0.12) * Math.min(1, (t - 1500) / overMs));

let fail = 0;
const expect = (name, ev, ok) => { const pass = ok(ev); if (!pass) fail++; console.log(`${pass ? 'ok  ' : 'FAIL'} ${name.padEnd(46)} ${ev.join(' ') || '(nothing)'}`); };
const confirms = (ev) => ev.includes('beep') && ev.some((e) => e.startsWith('long'));
const silent = (ev) => !ev.includes('beep') && !ev.some((e) => e.startsWith('long'));

for (const fps of [24, 30, 60]) {
  for (const closeMs of [80, 250, 400, 600]) {
    expect(`${fps}fps deliberate: close ${closeMs}ms, hold 500ms`, run(closure(closeMs, 500), { fps }), confirms);
  }
  expect(`${fps}fps slow close 500ms, weak eye at 60%`, run(closure(500, 500), { fps, weak: 0.6 }), confirms);
  expect(`${fps}fps natural blink (60ms close, 120ms)`, run(closure(60, 120), { fps }), silent);
  expect(`${fps}fps natural blink (80ms close, 200ms)`, run(closure(80, 200), { fps }), silent);
  expect(`${fps}fps looking down: lids creep to 0.40`, run(ramp(0.40, 600), { fps }), silent);
  expect(`${fps}fps leaning in: lids creep to 0.45 over 2s`, run(ramp(0.45, 2000), { fps }), silent);
}
console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);
