// Runs the REAL calibrate() in public/gaze.js end to end, in node, on a virtual clock. The camera,
// the face model and the page are stubs; the synthetic face looks wherever the dot is.
// Why: gaze-smoke.mjs only tests the math, and a shadowed variable in calibrate() once made every
// calibration throw on the first moving-dot frame while both smoke tests passed.
// Run: node eval/calibrate-smoke.mjs   (GAZE_JS=path/to/gaze.js tests another copy)
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const pub = resolve(import.meta.dirname, '../public');
const gazeUrl = pathToFileURL(resolve(process.env.GAZE_JS ?? `${pub}/gaze.js`)).href;

// The page imports by absolute URL path ('/gazemodel.js'). Point those at public/, and the vendored
// MediaPipe bundle at a stub whose detector reads globalThis.__face.
const stub = `export const FilesetResolver = { forVisionTasks: async () => ({}) };
export const FaceLandmarker = { createFromOptions: async () => ({ detectForVideo: () => globalThis.__face(), close() {} }) };`;
const hooks = `
const map = { '/gazemodel.js': ${JSON.stringify(pathToFileURL(`${pub}/gazemodel.js`).href)},
  '/blinkgate.js': ${JSON.stringify(pathToFileURL(`${pub}/blinkgate.js`).href)},
  '/vendor/vision_bundle.mjs': 'data:text/javascript,' + encodeURIComponent(${JSON.stringify(stub)}) };
export async function resolve(spec, ctx, next) {
  return map[spec] ? { url: map[spec], shortCircuit: true } : next(spec, ctx);
}`;
register('data:text/javascript,' + encodeURIComponent(hooks));

// ---- virtual clock: setTimeout, setInterval, requestAnimationFrame and performance.now ----
let now = 0, seq = 0;
const timers = new Map();
const add = (fn, ms, every) => { const id = ++seq; timers.set(id, { fn, at: now + Math.max(0, ms || 0), every }); return id; };
globalThis.setTimeout = (fn, ms) => add(fn, ms);
globalThis.setInterval = (fn, ms) => add(fn, ms, ms || 1);
globalThis.clearTimeout = globalThis.clearInterval = (id) => timers.delete(id);
globalThis.requestAnimationFrame = (fn) => add(() => fn(now), 1000 / 60);
Object.defineProperty(globalThis, 'performance', { value: { now: () => now }, configurable: true });
const flush = () => new Promise((r) => setImmediate(r));
async function runUntil(done, limitMs) {
  while (!done() && now < limitMs) {
    let next = null;
    for (const [id, t] of timers) if (!next || t.at < next[1].at || (t.at === next[1].at && id < next[0])) next = [id, t];
    if (!next) { await flush(); continue; }
    const [id, t] = next;
    now = Math.max(now, t.at);
    if (t.every) t.at = now + t.every; else timers.delete(id);
    t.fn();
    await flush();
  }
}

// ---- the page: one 1500x900 window, a 30 fps camera ----
const W = 1500, H = 900;
globalThis.window = { innerWidth: W, innerHeight: H, outerWidth: W, outerHeight: H + 80, screenX: 0, screenY: 0,
  screen: { width: 1512, height: 982 }, devicePixelRatio: 2 };
const video = { play: async () => {}, remove() {}, readyState: 4, videoWidth: 1280, videoHeight: 720,
  get currentTime() { return Math.floor(now / 33.3) * 0.0333; } };
globalThis.document = { hidden: false, addEventListener() {}, removeEventListener() {}, querySelector: () => null,
  createElement: (tag) => (tag === 'video' ? video : { getContext: () => ({}) }) };
const track = { label: 'Test camera', stop() {}, addEventListener() {},
  getSettings: () => ({ width: 1280, height: 720, frameRate: 30, deviceId: 'cam-1' }) };
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { mediaDevices: {
  getUserMedia: async () => ({ getVideoTracks: () => [track], getTracks: () => [track] }) } } });

// ---- the synthetic face (as in gaze-smoke.mjs): the irises shift with where he looks ----
let rs = 11;
const rnd = () => ((rs = (rs * 16807) % 2147483647) / 2147483647);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const base = Array.from({ length: 478 }, () => ({ x: 0.35 + 0.3 * rnd(), y: 0.35 + 0.3 * rnd(), z: 0.02 * gauss() }));
let look = { x: 0.5, y: 0.5 }, irisNoise = 0.0004;
const lids = [{ categoryName: 'eyeBlinkLeft', score: 0.05 }, { categoryName: 'eyeBlinkRight', score: 0.05 }];
globalThis.__face = () => ({
  faceBlendshapes: [{ categories: lids }],
  faceLandmarks: [base.map((p, i) => {
    let x = p.x + 0.0005 * gauss(), y = p.y;
    if (i >= 468) { x += (look.x - 0.5) * 0.012 + irisNoise * gauss(); y += (look.y - 0.5) * 0.007 + irisNoise * gauss(); }
    return { x: x + 0.0004 * gauss(), y: y + 0.0004 * gauss(), z: p.z };
  })],
});

const { createGaze } = await import(gazeUrl);
const events = [], errors = [];
const gaze = createGaze({
  onGaze() {}, onBlink() {}, onFace() {}, onError: (m) => errors.push(m),
  onCalibrationProgress: (p) => { events.push(p); if (p.x != null) look = { x: p.x, y: p.y }; },
});

let failed = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failed++; };

let started;
gaze.start().then((r) => { started = r; });
await runUntil(() => started !== undefined, 30000);
check(started === true, `camera + face model start (errors: ${errors.join(' | ') || 'none'})`);

async function calibrate() {
  const samples = [];
  let result, error;
  gaze.calibrate({ onSample: (n) => samples.push(n) }).then((r) => { result = r; }, (e) => { error = e; });
  await runUntil(() => result !== undefined || error !== undefined, now + 120000);
  return { result, error, samples, done: events.filter((e) => e.state === 'done').at(-1) };
}

// 1. A clean face: calibration finishes, the counter climbs, the map is taken.
const a = await calibrate();
check(!a.error, `calibration runs to the end${a.error ? `: threw "${a.error.message}"` : ''}`);
check(a.samples.length > 100 && a.samples.at(-1) > a.samples[0], `onSample counter climbs (${a.samples[0]} -> ${a.samples.at(-1)})`);
check(a.result === true && a.done?.usable === true && a.done?.kept === false,
  `good fit is used (usable=${a.done?.usable}, kept=${a.done?.kept}, err ${a.done?.errX?.toFixed(0)}x${a.done?.errY?.toFixed(0)} px)`);
check(gaze.calibrated, 'gaze is calibrated');

// 2. A noisy face: the fit comes out loose, and the working map from run 1 is kept.
irisNoise = 0.008;
events.length = 0;
const b = await calibrate();
check(!b.error, `second calibration runs to the end${b.error ? `: threw "${b.error.message}"` : ''}`);
check(b.result === false && b.done?.usable === false && b.done?.kept === true,
  `loose fit keeps the old map (usable=${b.done?.usable}, kept=${b.done?.kept}, err ${b.done?.errX?.toFixed(0)}x${b.done?.errY?.toFixed(0)} px)`);

console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
process.exit(failed ? 1 : 0);
