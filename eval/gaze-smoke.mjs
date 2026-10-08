// Guards the pure gaze math (public/gazemodel.js) against silent regression. Synthetic face:
// the irises shift with where he looks, plus head jitter and landmark noise. The fit must
// recover the screen point and the planted eye+camera delay. Run: node eval/gaze-smoke.mjs
// (eval/gaze/parity.mjs runs the same code over a real recording.)
import { makeReference, rawFeatures, makeLidBasis, featureVector, fitCalibration } from '../public/gazemodel.js';

let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const base = Array.from({ length: 478 }, () => ({ x: 0.35 + 0.3 * rnd(), y: 0.35 + 0.3 * rnd(), z: 0.02 * gauss() }));
const W = 1500, H = 900, LAG = 200;

function face(gx, gy) {   // gx, gy in [0,1]: where he is looking
  const yaw = 0.01 * gauss(), dx = 0.004 * gauss();
  return base.map((p, i) => {
    let x = p.x + dx + yaw * (p.z * 10), y = p.y;
    if (i >= 468) { x += (gx - 0.5) * 0.012; y += (gy - 0.5) * 0.007; }
    return { x: x + 0.0004 * gauss(), y: y + 0.0004 * gauss(), z: p.z };
  });
}
const still = [], moving = [], frames = [];
for (const [gx, gy] of [[0.1, 0.1], [0.5, 0.1], [0.9, 0.1], [0.1, 0.5], [0.5, 0.5], [0.9, 0.5], [0.1, 0.9], [0.5, 0.9], [0.9, 0.9]]) {
  for (let k = 0; k < 30; k++) still.push({ lm: face(gx, gy), x: gx * W, y: gy * H });
}
const dot = (t) => [0.5 + 0.45 * Math.sin((2 * Math.PI * t) / 13000), 0.5 + 0.45 * Math.sin((2 * Math.PI * t) / 8500 + 0.7)];
for (let t = 0; t < 35000; t += 33) { const [gx, gy] = dot(t - LAG); moving.push({ lm: face(gx, gy), t }); }

const cw = 1280, ch = 720;
const ref = makeReference([...still, ...moving].map((s) => s.lm), cw, ch);
const sr = still.map((s) => rawFeatures(s.lm, cw, ch, ref)), mr = moving.map((m) => rawFeatures(m.lm, cw, ch, ref));
const lb = makeLidBasis([...sr, ...mr].map((r) => r.lids));
const cal = fitCalibration(
  still.map((s, i) => ({ f: featureVector(sr[i], lb), x: s.x, y: s.y })),
  moving.map((m, i) => ({ f: featureVector(mr[i], lb), t: m.t })),
  (t) => { const [x, y] = dot(t); return [x * W, y * H]; },
);
const test = [[0.3, 0.3], [0.7, 0.6], [0.2, 0.8]].map(([gx, gy]) => {
  const [x, y] = cal.model.predict(featureVector(rawFeatures(face(gx, gy), cw, ch, ref), lb));
  return Math.hypot(x - gx * W, y - gy * H);
});
const worst = Math.max(...test);
console.log(`delay found ${cal.lag} ms (planted ${LAG}), held-out error ${test.map(Math.round).join('/')} px`);
if (!(worst < 60) || Math.abs(cal.lag - LAG) > 50) { console.error('FAIL'); process.exit(1); }
console.log('PASS');
