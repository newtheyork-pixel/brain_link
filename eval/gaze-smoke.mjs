// Guards the pure gaze math against silent regression — ridgeSolve/dotv/buildModel were once
// deleted and their callers left behind, so calibrate() threw ReferenceError and gaze was dead
// at HEAD with no test to catch it. This runs the math in Node (no browser, no MediaPipe).
import { readFile } from 'node:fs/promises';

const src = await readFile(new URL('../public/gaze.js', import.meta.url), 'utf8');
// pull the three pure functions + N_RBF out of the module (they use no browser globals)
const grab = (name, kind = 'function') => {
  const re = kind === 'function'
    ? new RegExp(`\\nfunction ${name}\\b`) : new RegExp(`\\n(?:const|let) ${name}\\b`);
  const start = src.search(re);
  if (start < 0) throw new Error(`MISSING: ${name} — gaze.js will throw at runtime`);
  // brace-match from the first { after the name
  let i = src.indexOf('{', start), depth = 0, j = i;
  for (; j < src.length; j++) { if (src[j] === '{') depth++; else if (src[j] === '}' && --depth === 0) { j++; break; } }
  if (kind === 'object') {
    let bi = src.indexOf('{', start), d = 0, e = bi;
    for (; e < src.length; e++) { if (src[e] === '{') d++; else if (src[e] === '}' && --d === 0) { e++; break; } }
    return src.slice(start, e) + ';';
  }
  return src.slice(start, kind === 'function' ? j : src.indexOf('\n', start));
};
const code = [
  'const N_RBF = 5;',
  'const dotv = (w,v)=>{let s=0;for(let i=0;i<w.length;i++)s+=w[i]*v[i];return s;};',
  grab('ridgeSolve'), grab('makeBasis'),
  grab('X_VARIANTS', 'object'), grab('Y_VARIANTS', 'object'),
  grab('fitAxis'), grab('buildModel'),
  'globalThis.__gaze = { buildModel, dotv };',
].join('\n');
globalThis.window = { innerWidth: 1512, innerHeight: 828 };
new Function(code)();
const { buildModel } = globalThis.__gaze;

// synthetic linear ground truth: screen_x = 1200*ix + 40, screen_y = 900*iy + 30
// realistic: iris moves ~±0.08 horiz / ~±0.03 vert across the screen, with per-frame noise
let seed = 1;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
const mk = (n) => Array.from({ length: n }, (_, k) => {
  const u = ((k * 37) % 100) / 100, v = ((k * 61) % 100) / 100;   // screen fractions 0..1
  const ix = -0.08 * (u - 0.5) * 2 + rnd() * 0.004;
  const iy = 0.03 * (v - 0.5) * 2 + rnd() * 0.004;
  return { ix, iy, yaw: rnd() * 0.02, pitch: rnd() * 0.02,
    ap: 0.3 - 0.1 * v, nx: 0, ny: 0, sc: 0, x: u * 1512, y: v * 828 };
});
const train = mk(140), test = mk(60);
const m = buildModel(train, test);
let err = 0;
for (const s of test) { const [x, y] = m.predict(s); err += Math.hypot(x - s.x, y - s.y); }
err /= test.length;

const ok = Number.isFinite(err) && err < 15;
console.log(`  buildModel ran. held-out X±${m.errX} Y±${m.errY} · predict mean err ${err.toFixed(1)}px`);
console.log(ok ? '  SMOKE TEST PASS' : '  SMOKE TEST FAIL');
process.exit(ok ? 0 : 1);
