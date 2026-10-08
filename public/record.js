// Eye recording for OFFLINE tuning of the tracker.
//
// The tracker was tuned live, one guess per calibration, on his face. That cannot converge: every
// run mixes the change with a new posture, new light and a new calibration. This page records
// one session of raw signal with known targets, and every model change is then measured against
// the same recording (eval/gaze/), the way any other model in this repo is evaluated.
//
// Per frame: all 478 landmarks, the eye blendshapes, the head matrix, a small grayscale crop of
// each eye, and where the dot was. Nothing leaves this machine: it is posted to the local server
// and written to data/recordings/ (gitignored).

import { FaceLandmarker, FilesetResolver } from '/vendor/vision_bundle.mjs';

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const shuffle = (a) => { a = [...a]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

const ID = new Date().toISOString().replace(/[:.]/g, '-');
const CROP_W = 64, CROP_H = 32;
// Eye contours (MediaPipe face mesh) — the crop box is fitted to these.
const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const BS_KEEP = ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft',
  'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight'];

let video, landmarker, cam = {};
let target = null, phase = 'idle';      // what he is meant to be looking at, in [0,1] screen coords
let buf = [], frames = 0, stopped = false, lastVt = -1;
const crop = document.createElement('canvas');
crop.width = CROP_W; crop.height = CROP_H;
const cctx = crop.getContext('2d', { willReadFrequently: true });

async function flush() {
  if (!buf.length) return;
  const lines = buf; buf = [];
  try {
    await fetch('/api/gazerec', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ id: ID, lines }) });
  } catch { buf = lines.concat(buf); }   // server hiccup: keep it and retry on the next flush
}

function eyeCrop(lm, idx) {
  const W = video.videoWidth, H = video.videoHeight;
  const xs = idx.map((i) => lm[i].x * W), ys = idx.map((i) => lm[i].y * H);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const w = (Math.max(...xs) - Math.min(...xs)) * 1.5, h = w / 2;
  cctx.drawImage(video, cx - w / 2, cy - h / 2, w, h, 0, 0, CROP_W, CROP_H);
  const d = cctx.getImageData(0, 0, CROP_W, CROP_H).data;
  const g = new Uint8Array(CROP_W * CROP_H);
  for (let i = 0; i < g.length; i++) g[i] = (d[i * 4] * 77 + d[i * 4 + 1] * 150 + d[i * 4 + 2] * 29) >> 8;
  let s = ''; for (let i = 0; i < g.length; i++) s += String.fromCharCode(g[i]);
  return { img: btoa(s), box: [Math.round(cx), Math.round(cy), Math.round(w)] };
}

let faceSeen = 0;
// Poll on every animation frame, like gaze.js. requestVideoFrameCallback never fires for a video
// that is not in the page, which left this stuck on "Looking for your face".
let lastErr = '';
function onFrame() {
  if (stopped) return;
  requestAnimationFrame(onFrame);
  if (video.readyState < 2) return;
  const vt = video.currentTime;
  if (vt === lastVt) return;
  lastVt = vt;
  const meta = null;
  const t = performance.now();
  let out;
  try { out = landmarker.detectForVideo(video, t); }
  catch (e) { if (String(e) !== lastErr) { lastErr = String(e); $('#status').textContent = `face model error: ${lastErr}`; } return; }
  const lm = out.faceLandmarks?.[0];
  if (!lm || lm.length < 478) { faceSeen = 0; return; }
  faceSeen++;
  if (phase === 'idle') return;

  const flat = new Array(lm.length * 3);
  for (let i = 0; i < lm.length; i++) {
    flat[i * 3] = Math.round(lm[i].x * 1e5); flat[i * 3 + 1] = Math.round(lm[i].y * 1e5); flat[i * 3 + 2] = Math.round(lm[i].z * 1e5);
  }
  const bs = {};
  for (const c of out.faceBlendshapes?.[0]?.categories ?? []) if (BS_KEEP.includes(c.categoryName)) bs[c.categoryName] = +c.score.toFixed(4);
  const m = out.facialTransformationMatrixes?.[0]?.data;
  const L = eyeCrop(lm, LEFT_EYE), R = eyeCrop(lm, RIGHT_EYE);
  buf.push({
    k: 'f', t: +t.toFixed(1), cap: meta?.captureTime ? +meta.captureTime.toFixed(1) : null, vt: +vt.toFixed(4),
    ph: phase, tx: target?.[0] ?? null, ty: target?.[1] ?? null,
    lm: flat, bs, m: m ? Array.from(m, (v) => +v.toFixed(5)) : null,
    eL: L.img, eR: R.img, bL: L.box, bR: R.box,
  });
  frames++;
  $('#status').textContent = `${frames} frames`;
  if (buf.length >= 60) flush();
}

const dot = $('#dot');
function show(x, y, shrink) {
  dot.style.display = 'block';
  dot.style.left = `${x * innerWidth}px`; dot.style.top = `${y * innerHeight}px`;
  dot.classList.remove('shrink');
  if (shrink) { void dot.offsetWidth; dot.classList.add('shrink'); }
}
function hide() { dot.style.display = 'none'; }

async function message(text, ms) {
  hide(); target = null;
  const m = $('#msg'); m.innerHTML = `<strong>${text}</strong>`; m.style.display = 'flex';
  await sleep(ms);
  m.style.display = 'none';
}

async function fixations(name, pts, holdMs) {
  phase = name;
  for (const [x, y] of pts) {
    if (stopped) return;
    target = [x, y]; show(x, y, true);
    await sleep(holdMs);
  }
}

async function pursuit(name, seconds) {
  phase = name;
  const t0 = performance.now();
  // A Lissajous figure: covers the whole screen, always moving, never a sharp turn the eye has to
  // catch up on. Peak speed ~350 px/s on a laptop, inside what smooth pursuit can follow.
  while (!stopped) {
    const s = (performance.now() - t0) / 1000;
    if (s > seconds) break;
    const x = 0.5 + 0.43 * Math.sin((2 * Math.PI * s) / 13);
    const y = 0.5 + 0.40 * Math.sin((2 * Math.PI * s) / 8.5 + 0.7);
    target = [x, y]; show(x, y, false);
    await new Promise((r) => requestAnimationFrame(r));
  }
}

const GRID = [];
for (const y of [0.08, 0.36, 0.64, 0.92]) for (const x of [0.06, 0.28, 0.5, 0.72, 0.94]) GRID.push([x, y]);
const NINE = [];
for (const y of [0.15, 0.5, 0.85]) for (const x of [0.12, 0.5, 0.88]) NINE.push([x, y]);
const TILES = [];
for (const y of [0.25, 0.75]) for (const x of [0.125, 0.375, 0.625, 0.875]) TILES.push([x, y]);

async function run() {
  $('#msg').innerHTML = '<strong>Loading the face model…</strong>';
  try { await document.documentElement.requestFullscreen(); } catch {}
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: 'user' } });
  video = document.createElement('video');
  video.muted = true; video.playsInline = true; video.srcObject = stream;
  await video.play();
  const files = await FilesetResolver.forVisionTasks('/vendor');
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: '/vendor/face_landmarker.task', delegate },
    outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true, runningMode: 'VIDEO', numFaces: 1 });
  let backend = 'GPU';
  try { landmarker = await FaceLandmarker.createFromOptions(files, opts('GPU')); }
  catch { landmarker = await FaceLandmarker.createFromOptions(files, opts('CPU')); backend = 'CPU'; }
  const s = stream.getVideoTracks()[0].getSettings();
  cam = { w: s.width, h: s.height, fps: s.frameRate, label: stream.getVideoTracks()[0].label };
  onFrame();

  $('#msg').innerHTML = '<strong>Looking for your face…</strong><small id="seen"></small>';
  const t0 = performance.now();
  while (faceSeen < 15) {
    await sleep(100);
    const secs = (performance.now() - t0) / 1000;
    if (secs > 4) $('#seen').textContent = `camera ${video.videoWidth}x${video.videoHeight}, time ${video.currentTime.toFixed(1)}s, `
      + `backend ${backend}. ${lastErr || 'No face found yet: face the camera, more light.'}`;
  }
  await sleep(300);   // let fullscreen settle before the geometry is recorded
  buf.push({ k: 'meta', id: ID, backend, cam, crop: [CROP_W, CROP_H],
    win: { w: innerWidth, h: innerHeight, sx: screenX, sy: screenY, ow: outerWidth, oh: outerHeight },
    screen: { w: screen.width, h: screen.height }, dpr: devicePixelRatio,
    fullscreen: !!document.fullscreenElement, ua: navigator.userAgent, at: new Date().toISOString() });

  const steps = [
    () => message('Look at the black centre of each dot.', 2500),
    () => fixations('fix1', shuffle(GRID), 1600),
    () => message('Now follow the moving dot with your eyes.', 2500),
    () => pursuit('pursuit', 35),
    () => message('Dots again.', 2000),
    () => fixations('fix2', shuffle(GRID), 1600),
    () => message('Lean a little to your LEFT, and stay there.', 3500),
    () => fixations('lean_left', shuffle(NINE), 1600),
    () => message('Lean a little to your RIGHT.', 3500),
    () => fixations('lean_right', shuffle(NINE), 1600),
    () => message('Sit back, a bit further from the screen.', 3500),
    () => fixations('far', shuffle(NINE), 1600),
    () => message('Come a bit closer to the screen.', 3500),
    () => fixations('near', shuffle(NINE), 1600),
    () => message('Back to normal. Last part: the eight tile positions.', 3500),
    () => fixations('tiles', shuffle([...TILES, ...TILES]), 2000),
  ];
  for (let i = 0; i < steps.length && !stopped; i++) {
    $('#bar').style.width = `${(100 * i) / steps.length}%`;
    await steps[i]();
  }
  finish();
}

async function finish() {
  if (stopped) return;
  stopped = true; phase = 'idle'; hide();
  buf.push({ k: 'end', frames, at: new Date().toISOString() });
  await flush();
  try { await document.exitFullscreen(); } catch {}
  document.body.style.cursor = 'default';
  const m = $('#msg');
  m.innerHTML = `<strong>Done. ${frames} frames saved.</strong><small>data/recordings/${ID}.jsonl. You can close this tab.</small>`;
  m.style.display = 'flex';
  video?.srcObject?.getTracks().forEach((t) => t.stop());
}

addEventListener('keydown', (e) => { if (e.key === 'Escape') finish(); });
$('#go').onclick = () => run().catch((e) => {
  $('#msg').innerHTML = `<strong>Could not start: ${e.message}</strong>`;
});
// Leaving fullscreen changes the screen geometry under the targets: stop rather than record lies.
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement && phase !== 'idle') finish(); });
