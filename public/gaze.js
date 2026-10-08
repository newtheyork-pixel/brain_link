// Eye tracking, from the plain webcam. Fully on-device.
//
// MediaPipe FaceLandmarker (vendored in /vendor — nothing is fetched at runtime) gives us 478
// face landmarks and 52 blendshapes per frame, including where each eye is pointing and how
// closed each lid is. We turn that into a gaze point on the screen.
//
// This is TIER 1 of the input ladder: the camera. It is the tier a real ALS user would choose
// for as long as it works — and the tier that quits on them first, when a drooping eyelid hides
// the pupil. That is what the electrodes are for later. Same six events either way.
//
// The signal is deliberately coarse. Webcam gaze is good to a few degrees, not a few pixels —
// so we never ask it to hit a small target. Eight big tiles, and a dwell to confirm.

import { FaceLandmarker, FilesetResolver } from '/vendor/vision_bundle.mjs';
import { RIGID, LEFT_EYE, RIGHT_EYE, makeReference, rawFeatures, makeLidBasis, featureVector,
  fitCalibration, fitRidge, makeMedian, makeFixation } from '/gazemodel.js';
import { makeBlinkGate, BLINK_DEFAULTS } from '/blinkgate.js';

// Only these landmarks feed the model; calibration keeps copies of just these, not all 478.
const NEED = [...new Set([...RIGID, ...LEFT_EYE, ...RIGHT_EYE, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477])];
const keepLm = (lm) => { const o = []; for (const i of NEED) o[i] = { x: lm[i].x, y: lm[i].y, z: lm[i].z }; return o; };

// Blinks are judged in blinkgate.js: when the lids start to move (gaze freezes), when both eyes
// count as shut, and how long. Two deliberate-blink lengths, so blinks can both NAVIGATE and CONFIRM:
//   short  (a quick, decided blink)        -> move to the next option
//   long   (eyes held shut >= confirmMs)   -> say the highlighted option
// Natural blinks on the 2026-10-08 recording measured 83-292 ms shut; confirmMs defaults to 350.
const BLINK_ON = 0.6;              // calibration and recenter still drop any frame this closed
// The iris landmarks. This model returns 478 points, and the last ten are the two irises —
// the actual dark circles of his eyes, tracked directly.
const IRIS_L = 468, IRIS_R = 473;
// Eye corners and lids, to measure WHERE IN THE EYE the iris is sitting.
const L_OUT = 33, L_IN = 133, L_TOP = 159, L_BOT = 145;
const R_IN = 362, R_OUT = 263, R_TOP = 386, R_BOT = 374;

/**
 * Where the eyes are pointing, plus where the head is pointing.
 *
 * THE FIX for "it just moves forever": I was reading the BLENDSHAPES (eyeLookIn/Out/Up/Down).
 * Those exist to animate cartoon avatars. They are coarse, heavily smoothed, and quantised —
 * fine for making a puppet glance sideways, hopeless for telling which of eight tiles a man is
 * looking at. Feeding them into a linear fit produced a dot that wandered and never settled.
 *
 * The same model also gives the IRIS LANDMARKS. So measure the thing directly: where does the
 * iris sit between the corners of the eye, and between the lids? That is a real, continuous,
 * high-resolution gaze signal — and it is what actual eye trackers use.
 *
 * We keep the head pose too. People aim with the eyes AND the head; the calibration fit works
 * out the mix for this person, in this chair.
 */
function features(lm, face, matrix) {
  const b = {};
  for (const c of face) b[c.categoryName] = c.score;

  // MEASURE IN 3D, IN THE EYE'S OWN FRAME.
  //
  // The 2D version measured the iris offset in IMAGE coordinates. But the iris sits in front of
  // the plane of the eye corners — so when the head yaws, perspective shifts the projected iris
  // relative to the corner midpoint even when the gaze has not moved. That parallax is exactly
  // the +0.023 DC offset that wrecked the live test after a head-still calibration.
  //
  // MediaPipe gives every landmark a z. So build each eye's own 3D frame — its x-axis along the
  // eye corners, its y-axis from the face's own vertical (forehead→chin, rigid points) — and
  // measure the iris offset along THOSE axes. The frame rotates with the head, so a head turn
  // cancels out geometrically instead of having to be learned statistically.
  const P = (i) => lm[i];
  const sub = (a, c) => ({ x: a.x - c.x, y: a.y - c.y, z: a.z - c.z });
  const dot3 = (a, c) => a.x * c.x + a.y * c.y + a.z * c.z;
  const len3 = (a) => Math.hypot(a.x, a.y, a.z);
  const scale3 = (a, k) => ({ x: a.x * k, y: a.y * k, z: a.z * k });
  const norm3 = (a) => scale3(a, 1 / (len3(a) || 1e-9));

  const irisCentre = (a, z) => {
    let x = 0, y = 0, zz = 0;
    for (let i = a; i <= z; i++) { x += lm[i].x; y += lm[i].y; zz += lm[i].z; }
    const n = z - a + 1;
    return { x: x / n, y: y / n, z: zz / n };   // the 5-point ring, not one point: ~5x less jitter
  };

  // The face's own vertical: forehead (10) to chin (152). Rigid bone landmarks — they do not
  // move when he blinks, talks, or squints, and they rotate with the head.
  const faceDown = norm3(sub(P(152), P(10)));

  const eye = (irisA, irisZ, c1, c2) => {
    const a = P(c1), c = P(c2);
    const ex = norm3(sub(c, a));                                   // along the eye, in 3D
    const ey = norm3(sub(faceDown, scale3(ex, dot3(faceDown, ex)))); // face-vertical ⊥ ex
    const w = len3(sub(c, a)) || 1e-6;
    const mid = { x: (a.x + c.x) / 2, y: (a.y + c.y) / 2, z: (a.z + c.z) / 2 };
    const d = sub(irisCentre(irisA, irisZ), mid);
    return { x: dot3(d, ex) / w, y: dot3(d, ey) / w };
  };

  // Corner order chosen so both eyes' x-axes point the same way across the face — the 2D version
  // got this wrong once and the two eyes cancelled each other to zero.
  const L = eye(468, 472, L_OUT, L_IN);
  const R = eye(473, 477, R_IN, R_OUT);
  const ix = (L.x + R.x) / 2;
  const iy = (L.y + R.y) / 2;

  // Lid opening in 3D, normalised by eye width. Looking down closes the lid; without this the
  // model cannot separate a lowered eye from a lowered eyelid.
  const apOf = (top, bot, c1, c2) =>
    len3(sub(P(bot), P(top))) / (len3(sub(P(c2), P(c1))) || 1e-6);
  const ap = (apOf(L_TOP, L_BOT, L_OUT, L_IN) + apOf(R_TOP, R_BOT, R_IN, R_OUT)) / 2;

  // Head yaw/pitch out of the 4x4 rigid transform (column-major).
  let yaw = 0, pitch = 0;
  if (matrix) {
    const m = matrix.data;
    yaw = Math.atan2(-m[8], Math.hypot(m[9], m[10]));
    pitch = Math.atan2(m[9], m[10]);
  }

  // WHERE THE HEAD IS, not just where it points. Sliding sideways in the chair changes the
  // camera's viewing angle onto the eye without changing yaw at all — translation needs its own
  // features or it becomes an unexplained offset. Nose tip for position, inter-ocular distance
  // for range.
  const nose = P(1);
  const nx = nose.x - 0.5, ny = nose.y - 0.5;
  const sc = Math.log(len3(sub(P(L_OUT), P(R_OUT))) || 1e-6);

  const lidL = b.eyeBlinkLeft ?? 0, lidR = b.eyeBlinkRight ?? 0;
  return { v: [ix, iy, yaw, pitch, ap, nx, ny, sc], lid: Math.max(lidL, lidR), L: lidL, R: lidR };
}

// THE OUTPUT STAGE (median + fixation lock) lives in gazemodel.js, so the offline blink replay
// (eval/gaze/blink.mjs) runs the same code the cursor does.

// getUserMedia can hang forever: no camera, or one another app is holding. A promise that never
// settles means no error, no toast, and a user tapping a button that says nothing back. Every
// await on the camera gets a deadline.
const deadline = (p, ms, what) => Promise.race([
  p,
  new Promise((_, rej) => setTimeout(() => rej(new Error(`${what} timed out`)), ms)),
]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Open the front camera and a playing <video> on it. Throws on failure; cleans up after itself. */
async function openCamera() {
  // 640x480 leaves the iris about ten pixels across at laptop distance — the landmark then
  // quantises to that grid and the jitter IS the signal. Ask for everything the camera has.
  const s = await deadline(
    navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280, min: 640 },
        height: { ideal: 720, min: 480 },
        frameRate: { ideal: 30 },
        facingMode: 'user',
      },
    }),
    12000, 'camera',
  );
  const v = document.createElement('video');
  v.autoplay = true; v.playsInline = true; v.muted = true;
  v.srcObject = s;
  try { await deadline(v.play(), 5000, 'video'); }
  catch (e) { s.getTracks().forEach((t) => t.stop()); e.noFrames = true; throw e; }
  return { s, v };
}

// onError(msg, { fatal }): fatal means eye tracking has stopped and needs a person (camera blocked,
// no camera, model would not load). Anything else is a message; tracking carries on or recovers.
export function createGaze({ onGaze, onBlink, onBlinkHeld, onLidsClosing, onLids, onFace, onError,
  onRecovered, onCalibrationProgress, confirmMs = BLINK_DEFAULTS.confirmMs }) {
  let landmarker = null, video = null, stream = null, backend = '?';
  let camera = null, irisPx = 0;
  let running = false, calibrating = false, cancelled = false;
  let model = null;                       // the eye→screen map, once calibrated
  let fit = null;                         // { ref, lidBasis, lambda, lag, cw, ch } — what the model reads
  let bias = { x: 0, y: 0 };              // constant drift correction, from recenter()
  let lastF = null;                       // the latest SMOOTHED model features (14 numbers)

  // THE TERRAIN MAP. Every (features -> screen-point) pair we trust: the calibration at the start,
  // plus every real selection since. Real selections are few against thousands of calibration
  // frames, so each one counts LEARN_WEIGHT times in the refit — otherwise they could never move it.
  let terrain = [];
  const TERRAIN_CAP = 6000;               // keep it current: old samples fall off the front
  const REFIT_EVERY = 10;                 // refit after this many real selections
  const LEARN_WEIGHT = 20;
  let sinceRefit = 0, learnId = 0;
  let fitThroughId = 0;                   // learned rows up to this id are in the current model

  function rebuild() {
    if (!fit || terrain.length < 100) return false;
    const X = [], Y = [];
    for (const s of terrain) {
      const k = s.id ? LEARN_WEIGHT : 1;
      for (let j = 0; j < k; j++) { X.push(s.f); Y.push([s.x, s.y]); }
    }
    const next = fitRidge(X, Y, fit.lambda);
    // A refit must never replace a working map with a broken one.
    const probe = next.predict(terrain[terrain.length - 1].f);
    if (!probe.every(Number.isFinite)) return false;
    model = next;
    return true;
  }

  /** Model features for one frame, or null before calibration. */
  const modelFeatures = (lm) => featureVector(rawFeatures(lm, fit.cw, fit.ch, fit.ref), fit.lidBasis);
  let collector = null;                   // calibration sink: every camera frame, timestamped
  let median = makeMedian();
  let fixate = makeFixation();
  let lastTs = -1;

  // THE BLINK GATE. From the first frame the lids start to move until they are back where they
  // were, no frame reaches lastF, the median, the fixation lock or onGaze. preBlink is what the
  // tracker knew just before: the smoothed features and the point he was looking at. On release we
  // restore it, so the dot does not jump a row after every blink, and a blink-confirmed selection
  // learns from his open eye, not from a closing lid.
  const gate = makeBlinkGate({ confirmMs });
  let preBlink = null;        // { f, x, y, mx, my, locked, at }
  let lastOut = null;         // the last point handed to onGaze, with its median

  // SMOOTH THE FEATURES, NOT JUST THE OUTPUT.
  //
  // I was smoothing the dot AFTER the model had already amplified the noise. By then the damage
  // is done — you cannot un-amplify. The iris landmark jitters every frame (all of them do); it
  // has to be steadied BEFORE it reaches the model.
  let fsm = null;
  const smoothFeatures = (v) => {
    if (!fsm) { fsm = [...v]; return fsm; }
    for (let i = 0; i < v.length; i++) fsm[i] += 0.30 * (v[i] - fsm[i]);
    return [...fsm];
  };

  // How much does the raw signal wobble while he holds still? This is the number that decides
  // whether eye tracking is possible on this camera at all — and it was invisible until now.
  const noiseBuf = [];
  function trackNoise(v) {
    noiseBuf.push([v[0], v[1]]);
    if (noiseBuf.length > 60) noiseBuf.shift();
  }
  function featureNoise() {
    if (noiseBuf.length < 20) return null;
    const sd = (k) => {
      const a = noiseBuf.map((p) => p[k]);
      const m = a.reduce((x, y) => x + y, 0) / a.length;
      return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length);
    };
    return { irisX: +sd(0).toFixed(4), irisY: +sd(1).toFixed(4) };
  }
  // Both are null while no face is seen, and each carries a frame number (seq). Without that, a
  // poll after he turned away kept reading the last good frame: recenter "succeeded" on 55 copies
  // of one stale value, and the signal check measured zero noise on a frozen reading.
  let lastSample = null;      // the latest gaze point + raw features, for the accuracy test
  let lastRaw = null;         // { v, seq }: the unsmoothed features, open-eye frames only (signal check)
  let rawSeq = 0, gazeSeq = 0;
  let lastFrameAt = 0, detectFails = 0, stallTimer = null;
  let lastLid = 0;            // reject frames where he blinked: they carry no gaze at all
  let session = 0;            // bumped by stop(): a camera restart still in flight must not revive it
  let recovering = false;
  let importProblem = '';     // why the last import() refused a saved map, for the app to say

  // WHERE THE PAGE SITS ON THE GLASS. The map turns eyes into page pixels, but his eyes point at
  // the physical screen. When the page moves on the screen (fullscreen on or off, a toolbar, the
  // window dragged), every target moves by that much while his eyes do not. That is a shift, not
  // a scale, so the map is kept in the page coordinates of calibration time and shifted by how far
  // the page has moved since. Browser chrome is assumed to sit on top (Safari and Chrome on a Mac).
  function pageOrigin() {
    const cx = window.outerWidth - window.innerWidth, cy = window.outerHeight - window.innerHeight;
    // Page zoom or a docked panel makes the chrome size meaningless: then assume nothing moved.
    if (!(cx >= 0 && cx < 400 && cy >= 0 && cy < 400)) return null;
    return { x: window.screenX + cx / 2, y: window.screenY + cy,
      sw: window.screen.width, sh: window.screen.height, dpr: window.devicePixelRatio };
  }
  /** How far the page has moved on the screen since calibration, in page pixels. */
  function pageShift() {
    const a = fit?.origin, b = pageOrigin();
    // A different monitor or zoom level is not a shift we can work out: leave the map as it is.
    if (!a || !b || a.sw !== b.sw || a.sh !== b.sh || a.dpr !== b.dpr) return { x: 0, y: 0 };
    return { x: a.x - b.x, y: a.y - b.y };
  }

  /** Was this map made on the camera that is open now? A different lens or aspect moves every feature. */
  function sameCamera(f) {
    const c = f?.cam;
    // Either the id or the label matching is enough. Some browsers hand out a new deviceId after a
    // permission or site-data reset, and the same camera was then refused. Refuse only when every
    // name both sides have disagrees.
    const idSays = c?.id && camera?.id ? c.id === camera.id : null;
    const labelSays = c?.label && camera?.label ? c.label === camera.label : null;
    if ((idSays === false && labelSays !== true) || (labelSays === false && idSays !== true)) return false;
    if (f?.cw && f?.ch && camera?.w && camera?.h && Math.abs(f.cw / f.ch - camera.w / camera.h) > 0.02) return false;
    return true;
  }

  // A background tab gets no animation frames, so no camera frames are read. That is not a dead
  // camera: restart the stall clock when he comes back, or the first check fires on the old time.
  const onVisible = () => { if (!document.hidden) lastFrameAt = performance.now(); };

  function teardown() {
    running = false;
    recovering = false;
    session++;
    clearInterval(stallTimer);
    document.removeEventListener('visibilitychange', onVisible);
    stream?.getTracks().forEach((t) => t.stop());
    video?.remove();
    video = stream = null;
  }

  /** Make a freshly opened camera the live one. */
  function attach(s, v) {
    stream = s; video = v; lastTs = -1;
    const tr = stream.getVideoTracks()[0];
    const t = tr?.getSettings?.() ?? {};
    camera = { w: t.width ?? 0, h: t.height ?? 0, fps: t.frameRate ?? 0, id: t.deviceId ?? '', label: tr?.label ?? '' };
    // The track ending (unplugged, revoked, grabbed by another app) is recoverable too: it may
    // come back, and he has no hands to turn eye tracking off and on.
    const mine = stream;
    stream.getVideoTracks()[0]?.addEventListener('ended', () => {
      if (mine === stream) recover('The camera was disconnected.');
    });
  }

  /**
   * THE CAMERA HICCUPPED. A stall, a dropped track or a run of failed frames used to switch eye
   * tracking off for good while the setting still said Eyes, and he had no input at all until
   * someone noticed. Now: show the dot as lost, keep the calibrated map, and reopen the camera with
   * backoff until it comes back. Only a blocked camera (a permission, which needs a person) stops.
   */
  async function recover(why) {
    if (!running || recovering) return;
    recovering = true;
    const mine = session;
    onFace(false);
    gate.reset(); preBlink = null; lastOut = null;
    lastRaw = null; lastSample = null;
    onError(`${why} Restarting the camera...`, { fatal: false });
    stream?.getTracks().forEach((t) => t.stop());
    video?.remove();
    video = stream = null;              // loop() idles until the new camera is up
    for (let attempt = 0; ; attempt++) {
      await sleep(Math.min(15000, 1000 * 2 ** attempt));
      if (mine !== session) return;     // stopped while we waited
      try {
        if (!landmarker) await deadline(load(), 25000, 'face model');
        const { s, v } = await openCamera();
        if (mine !== session) { s.getTracks().forEach((t) => t.stop()); return; }
        attach(s, v);
        lastF = null; fsm = null;
        median = makeMedian();
        fixate = makeFixation();
        detectFails = 0;
        lastFrameAt = performance.now();
        recovering = false;
        onRecovered?.();
        // The camera that came back may be a different one (the external one unplugged, the
        // built-in one opened instead). The map still runs, but it was made for the other lens.
        if (fit && !sameCamera(fit)) onError('A different camera is on now. Calibrate again for this one.', { fatal: false });
        return;
      } catch (e) {
        if (mine !== session) return;
        if (e.name === 'NotAllowedError') {
          recovering = false;
          onError('Camera blocked. Allow it in the address bar, then choose Eyes again in Settings.', { fatal: true });
          return;
        }
        // Anything else (no camera yet, still busy, timed out): try again, a little later each time.
      }
    }
  }

  async function load() {
    // MediaPipe's vision graph needs a GL context for image conversion — with EITHER delegate.
    // Without one it dies deep inside the WASM with "Cannot read properties of undefined
    // (reading 'activeTexture')", which tells the user nothing. Check first, and say so plainly.
    const probe = document.createElement('canvas');
    if (!(probe.getContext('webgl2') || probe.getContext('webgl'))) {
      throw new Error('this browser has no WebGL — eye tracking needs it. Try Chrome with hardware acceleration on.');
    }

    const files = await FilesetResolver.forVisionTasks('/vendor');
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: '/vendor/face_landmarker.task', delegate },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: 'VIDEO',
      numFaces: 1,
    });
    // GPU is ~3x faster, but it hard-fails on machines without a usable WebGL context (and
    // headlessly). Falling back to CPU costs frames; refusing to run costs him the feature.
    try {
      landmarker = await FaceLandmarker.createFromOptions(files, opts('GPU'));
      backend = 'GPU';
    } catch {
      landmarker = await FaceLandmarker.createFromOptions(files, opts('CPU'));
      backend = 'CPU';
    }
  }

  function loop() {
    if (!running) return;
    requestAnimationFrame(loop);

    if (!video || recovering || video.readyState < 2 || video.currentTime === lastTs) return;
    lastTs = video.currentTime;
    lastFrameAt = performance.now();       // for the stall watchdog

    let out;
    try { out = landmarker.detectForVideo(video, performance.now()); detectFails = 0; }
    catch {
      // A camera that throws every frame would freeze the dot silently. After a run of failures,
      // reload the face model and reopen the camera rather than leave him staring at a dead cursor.
      if (++detectFails === 30) {
        try { landmarker?.close?.(); } catch {}
        landmarker = null;
        recover('Eye tracking stopped reading the camera.');
      }
      return;
    }

    const face = out.faceBlendshapes?.[0]?.categories;
    const lm = out.faceLandmarks?.[0];
    // 478 landmarks means the iris points are there. 468 means they are not, and gaze is dead.
    if (!face?.length || !lm || lm.length < 478) {
      gate.reset(); preBlink = null; lastRaw = null; lastSample = null;
      onFace(false);
      return;
    }
    onFace(true);

    // How many pixels wide is the iris, really? Below ~12 the landmark cannot resolve where it
    // is sitting, and no algorithm downstream can recover that.
    if (camera) {
      irisPx = Math.round(Math.hypot(lm[471].x - lm[469].x, lm[471].y - lm[469].y) * camera.w);
    }

    const fRaw = features(lm, face, out.facialTransformationMatrixes?.[0]);
    lastLid = fRaw.lid;
    trackNoise(fRaw.v);
    const f = { ...fRaw, v: smoothFeatures(fRaw.v) };

    // The lids, both eyes, against his own open level (blinkgate.js). The snapshot is taken BEFORE
    // onLidsClosing, so the app reads the point from before the lids moved.
    const g = gate.step(lastFrameAt, f.L, f.R);
    // Every frame, shut or not, so the app can show how long his eyes have been closed. A blink
    // that "did nothing" is then visibly "not long enough", not a mystery.
    onLids?.(gate.shut);
    // A blink frame carries no gaze. One in the signal check's "hold still" window used to turn a
    // good camera into "too noisy".
    lastRaw = g.gated || fRaw.lid > BLINK_ON ? null : { v: fRaw.v, seq: ++rawSeq };
    if (g.entered) {
      // Only a point from the last few frames counts: one from before a face loss is stale.
      preBlink = lastOut && lastF && lastFrameAt - lastOut.t < 200
        ? { ...lastOut, f: [...lastF], at: lastFrameAt } : null;
      onLidsClosing?.();
    }
    for (const e of g.events) {
      if (e.type === 'held') onBlinkHeld?.(e.held);
      else onBlink(e.kind, e.held);
    }

    // Calibration takes EVERY frame, stamped with when it was seen — the moving-dot labels are
    // matched to the dot's position a measured delay earlier, so the time has to be exact.
    if (collector) collector({ lm: keepLm(lm), t: lastFrameAt, lid: fRaw.lid, gated: g.gated });

    if (!model) return;

    // Lids moving: hold everything still. Otherwise the cursor lurches away every blink, and the
    // lurch un-arms the very tile the blink was meant to say.
    if (g.gated) return;

    const fv = modelFeatures(lm);
    if (preBlink) {
      // First frame after a blink: carry on from the open-eye state, not from half-closed lids.
      lastF = preBlink.f;
      median.fill(preBlink.mx, preBlink.my);
      preBlink = null;
    }
    lastF = lastF ? lastF.map((v, k) => v + 0.30 * (fv[k] - v)) : fv;   // steady BEFORE the model
    const [px0, py0] = model.predict(lastF);
    const sh = pageShift();
    const rx = Math.max(0, Math.min(window.innerWidth, px0 + bias.x + sh.x));
    const ry = Math.max(0, Math.min(window.innerHeight, py0 + bias.y + sh.y));

    const now = performance.now();
    const [mx, my] = median(rx, ry);
    const [x, y, locked] = fixate(mx, my, now);

    lastOut = { x, y, mx, my, locked, t: lastFrameAt };
    lastSample = { x, y, locked, raw: f.v.map((n) => +n.toFixed(3)), seq: ++gazeSeq };
    onGaze(x, y, locked);
  }

  return {
    get calibrated() { return !!model; },
    get running() { return running; },
    get backend() { return backend; },

    async start() {
      // Starting is the one place a failure is final: nothing was working yet, and a person has to
      // fix it (allow the camera, plug one in). Once running, hiccups recover on their own.
      const mine = session;
      let cam;
      try { cam = await openCamera(); }
      catch (e) {
        onError(e.name === 'NotAllowedError'
          ? 'Camera blocked. Allow it in the address bar, then choose Eyes again in Settings.'
          : e.name === 'NotFoundError' ? 'No camera found on this machine.'
          : e.noFrames ? `Camera opened but no frames: ${e.message}`
          : `Camera: ${e.message}`, { fatal: true });
        return false;
      }
      if (mine !== session) { cam.s.getTracks().forEach((t) => t.stop()); return false; }
      attach(cam.s, cam.v);

      try { if (!landmarker) await deadline(load(), 25000, 'face model'); }
      catch (e) { onError(`Eye tracking unavailable — ${e.message}`, { fatal: true }); teardown(); return false; }
      if (mine !== session) return false;

      running = true;
      lastFrameAt = performance.now();
      document.addEventListener('visibilitychange', onVisible);
      // Liveness watchdog: if frames stop arriving for ~2s while the page is in front, reopen the
      // camera. A hidden tab is skipped: it gets no frames by design.
      clearInterval(stallTimer);
      stallTimer = setInterval(() => {
        if (!running || recovering) return;
        if (document.hidden) { lastFrameAt = performance.now(); return; }
        if (performance.now() - lastFrameAt > 2000) {
          recover('Camera stopped sending frames. Check it is not covered or in use elsewhere.');
        }
      }, 1000);

      loop();
      return true;
    },

    /**
     * CALIBRATION: nine still dots, then follow a smoothly moving dot.
     *
     * Measured on a recording (eval/gaze/calib_design.py): still dots alone are too few, the moving
     * dot alone is only good once its labels are shifted by the eye+camera delay (~150-300 ms), and
     * both together are best. The still dots have no delay, so they also referee which delay and
     * which smoothing (lambda) to use — see fitCalibration.
     */
    async calibrate({ bounds, onSample } = {}) {
      if (!running || calibrating) return false;
      calibrating = true;
      cancelled = false;
      const frames = [];
      const ok = (fr) => !fr.gated && fr.lid < BLINK_ON;   // no frame with the lids moving
      let usable = 0;                                        // what the counter shows: frames the fit can use
      collector = (fr) => { frames.push(fr); if (ok(fr)) usable++; };
      try {
        const B = bounds ?? { x0: 0.10, x1: 0.90, y0: 0.16, y1: 0.86 };
        const W = window.innerWidth, H = window.innerHeight;
        const origin = pageOrigin();
        const lerp = (a, b, t) => a + (b - a) * t;
        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
        // A camera that drops out mid-run used to leave the dots running over no frames for up to
        // 45 s. Stop at the next step instead; the app says why and he can start again.
        const check = () => {
          if (cancelled) throw new Error('cancelled');
          if (!running || recovering) throw new Error('the camera dropped out. Try again when it is back.');
        };

        // 1. Still dots. Only frames from 500 ms after the dot lands count: before that the eye is
        //    still travelling.
        const dots = [];
        for (const fy of [0, 0.5, 1]) for (const fx of [0, 0.5, 1]) dots.push([lerp(B.x0, B.x1, fx), lerp(B.y0, B.y1, fy)]);
        for (let i = dots.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [dots[i], dots[j]] = [dots[j], dots[i]]; }
        const stillWin = [];
        for (let i = 0; i < dots.length; i++) {
          check();
          const [x, y] = dots[i];
          onCalibrationProgress({ state: 'point', x, y, index: i, total: dots.length });
          await sleep(500);
          check();
          onCalibrationProgress({ state: 'sampling', x, y, index: i, total: dots.length });
          const t0 = performance.now();
          await sleep(1100);
          stillWin.push({ t0, t1: performance.now(), x: x * W, y: y * H });
        }

        // 2. The moving dot: a Lissajous figure over the working area, no sharp turns to catch up on.
        const timeline = [];
        const SECS = 35;
        const at = (sec) => [lerp(B.x0, B.x1, 0.5 + 0.5 * Math.sin((2 * Math.PI * sec) / 13)),
          lerp(B.y0, B.y1, 0.5 + 0.5 * Math.sin((2 * Math.PI * sec) / 8.5 + 0.7))];
        // Wait where the path STARTS. Waiting at the top edge made the dot jump most of the area the
        // moment it began to move, and he lost it.
        const [sx, sy] = at(0);
        onCalibrationProgress({ state: 'pursuit', start: true, x: sx, y: sy, progress: 0 });
        await sleep(1500);
        const tStart = performance.now();
        const pursuitFrom = frames.length;
        for (;;) {
          check();
          const sec = (performance.now() - tStart) / 1000;
          if (sec > SECS) break;
          const [x, y] = at(sec);
          timeline.push({ t: performance.now(), x: x * W, y: y * H });
          onCalibrationProgress({ state: 'pursuit', x, y, progress: sec / SECS });
          onSample?.(usable);
          await new Promise((r) => requestAnimationFrame(r));
        }
        collector = null;

        const still = [];
        for (const w of stillWin) {
          for (const fr of frames) if (fr.t >= w.t0 && fr.t <= w.t1 && ok(fr)) still.push({ fr, x: w.x, y: w.y });
        }
        const moving = frames.slice(pursuitFrom).filter((fr) => ok(fr) && fr.t >= timeline[0].t + 400 && fr.t <= timeline.at(-1).t);
        // A plain throw, not onError: this is a result he can retry at once, with the camera on.
        if (still.length < 60 || moving.length < 200) {
          throw new Error('your face was not visible enough. More light, sit closer.');
        }

        const cw = video?.videoWidth || camera.w, ch = video?.videoHeight || camera.h;   // video is null mid-restart
        const ref = makeReference([...still.map((s) => s.fr.lm), ...moving.map((m) => m.lm)], cw, ch);
        const rawOf = (lm) => rawFeatures(lm, cw, ch, ref);
        const stillRaw = still.map((s) => rawOf(s.fr.lm)), movingRaw = moving.map((m) => rawOf(m.lm));
        const lidBasis = makeLidBasis([...stillRaw, ...movingRaw].map((r) => r.lids));
        const dotAt = (t) => {
          let lo = 0, hi = timeline.length - 1;
          if (t <= timeline[0].t) return [timeline[0].x, timeline[0].y];
          while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (timeline[mid].t <= t) lo = mid; else hi = mid; }
          const a = timeline[lo], b = timeline[hi], u = Math.min(1, (t - a.t) / ((b.t - a.t) || 1));
          return [a.x + u * (b.x - a.x), a.y + u * (b.y - a.y)];
        };
        const cal = fitCalibration(
          still.map((s, i) => ({ f: featureVector(stillRaw[i], lidBasis), x: s.x, y: s.y })),
          moving.map((m, i) => ({ f: featureVector(movingRaw[i], lidBasis), t: m.t })),
          dotAt,
        );

        const t = document.querySelector('.tile')?.getBoundingClientRect();
        const tileW = t?.width || W / 4, tileH = t?.height || H / 2;
        // Not called `usable`: that name is the frame counter above, and a const here would shadow it
        // for the whole try block, so the pursuit loop's onSample(usable) threw before this line ran.
        const fitOk = cal.errX < tileW * 0.45 && cal.errY < tileH * 0.45;

        // A LOOSE FIT NEVER REPLACES A WORKING ONE. He recalibrates while slumped, the fit comes out
        // too loose, and the map he has been using all week used to be gone, in memory and on disk 4 s
        // later. Keep the old one. With no map at all a loose one beats none, but it is marked loose
        // so it is never saved over a map on disk.
        const kept = !fitOk && !!model && !fit?.loose;
        if (!kept) {
          fit = { ref, lidBasis, lambda: cal.lambda, lag: cal.lag, cw, ch, errX: cal.errX, errY: cal.errY,
            origin, cam: { id: camera?.id ?? '', label: camera?.label ?? '' }, loose: !fitOk };
          model = cal.model;
          terrain = cal.samples.map((p) => ({ f: p.f, x: p.x, y: p.y }));
          sinceRefit = 0;
          bias = { x: 0, y: 0 };
          lastF = null; lastOut = null; preBlink = null;
          median = makeMedian();
          fixate = makeFixation();
        }

        onCalibrationProgress({
          state: 'done', errorPx: cal.stillErrPx, errX: cal.errX, errY: cal.errY,
          usable: fitOk, kept, backend,
          variant: `linear · delay ${cal.lag}ms · λ${cal.lambda}`, lag: cal.lag, lambda: cal.lambda,
          samples: still.length + moving.length, nStill: still.length, nMoving: moving.length,
          tile: { w: Math.round(tileW), h: Math.round(tileH) },
        });
        return !kept;
      } finally {
        // Whatever happens (a throw, a cancel, a lost face) the flag comes down. Otherwise a failed
        // run locks calibration out forever and he can never try again.
        collector = null;
        calibrating = false;
      }
    },

    get calibrating() { return calibrating; },
    cancelCalibration() { cancelled = true; },

    /**
     * LEARN FROM A REAL SELECTION. He dwelled inside a tile and confirmed it, so his gaze WAS at
     * that tile — a free, correctly-labelled training point, collected in the exact posture and
     * light he actually uses. Append it and, every so often, refit the whole map.
     *
     * Returns an id so the caller can RETRACT this sample if he immediately undoes the word — a
     * mis-selection he corrects is a poisoned label, and learning from it would make the map worse.
     */
    learn(feats, screenX, screenY) {
      // feats: the features to label, when the caller has better ones than "now". A blink commit
      // passes the pre-blink snapshot, because by the time the blink is over lastF is the reopen.
      const f = feats ?? lastF;
      if (!model || !f) return null;
      const id = ++learnId;
      // The tile is where it is on the page NOW; the terrain is in calibration-time page pixels.
      const sh = pageShift();
      terrain.push({ f: [...f], x: screenX - sh.x, y: screenY - sh.y, id });
      if (terrain.length > TERRAIN_CAP) terrain.shift();
      if (++sinceRefit >= REFIT_EVERY) { sinceRefit = 0; fitThroughId = learnId; rebuild(); }
      return id;
    },

    /** He undid the word — that selection was wrong, so drop what we "learned" from it. */
    retract(id) {
      if (id == null) return;
      const i = terrain.findIndex((s) => s.id === id);
      if (i < 0) return;
      terrain.splice(i, 1);
      // Already baked into the current map? Then dropping the row is not enough: refit without it.
      if (id <= fitThroughId) rebuild(); else sinceRefit = Math.max(0, sinceRefit - 1);
    },

    /** What the tracker knew just before the lids started to move (valid from onLidsClosing). */
    preBlink() { return preBlink ? { ...preBlink, f: [...preBlink.f] } : null; },

    /** How long the eyes must stay shut to count as a deliberate "say it". */
    get confirmMs() { return gate.confirmMs; },
    set confirmMs(ms) { gate.confirmMs = ms; },

    /** Persist / restore the calibration + terrain, so he never re-teaches the app his own eyes. */
    export() {
      // A loose map (made with no earlier one to fall back on) runs this session but is never
      // saved: it would overwrite whatever good map is on disk.
      return model && fit && !fit.loose ? { v: 3, fit, terrain, bias } : null;
    },
    import(saved) {
      importProblem = '';
      // v2 maps were made by the old tracker's features; they cannot drive this model.
      if (saved?.v !== 3 || !saved.fit?.ref || !Array.isArray(saved.terrain)) return false;
      // Made on another camera, or this one at a different shape: every feature is off, and the
      // "welcome back" toast would have told him all was well.
      if (!sameCamera(saved.fit)) { importProblem = 'camera'; return false; }
      const clean = saved.terrain.filter((s) => Array.isArray(s.f) && s.f.every(Number.isFinite)
        && Number.isFinite(s.x) && Number.isFinite(s.y));
      if (clean.length < 100) return false;
      // No rescaling by window size: a window that changed size or moved is a SHIFT of the page on
      // the screen (pageShift), not a stretch. A map saved before the origin was recorded is taken
      // to be where the page is now.
      fit = saved.fit.origin ? saved.fit : { ...saved.fit, origin: pageOrigin() };
      terrain = clean.slice(-TERRAIN_CAP);
      learnId = terrain.reduce((m, s) => (s.id > m ? s.id : m), 0);
      fitThroughId = learnId;               // import() refits on everything it loaded
      const b = saved.bias ?? { x: 0, y: 0 };
      bias = { x: Number.isFinite(b.x) ? b.x : 0, y: Number.isFinite(b.y) ? b.y : 0 };
      lastF = null; lastOut = null; preBlink = null;
      median = makeMedian();
      fixate = makeFixation();
      return rebuild();
    },
    get terrainSize() { return terrain.length; },
    get importProblem() { return importProblem; },

    recalibrate() { model = null; fit = null; bias = { x: 0, y: 0 }; },

    /**
     * ONLINE RE-ANCHORING — the highest-leverage 15 lines in this whole tracker.
     *
     * Every confirmed selection is free ground truth: he dwelled inside a tile, so his gaze was
     * AT that tile. The gap between the tile centre and where the tracker thought he was looking
     * is the current bias — posture shift, chair moved, headrest adjusted, afternoon light.
     * Nudge a fraction of it away on every selection and the DC-drift failure class is dead
     * PERMANENTLY, for any tracker, without him ever recalibrating.
     *
     * Clamped per step, so one mis-selection cannot yank the map; α keeps it a follower, not a
     * twitch.
     */
    nudge(dx, dy, alpha = 0.25) {
      if (!model) return;
      bias.x += Math.max(-120, Math.min(120, alpha * dx));
      bias.y += Math.max(-120, Math.min(120, alpha * dy));
    },
    get bias() { return { x: Math.round(bias.x), y: Math.round(bias.y) }; },

    /**
     * RECENTER. He has been sitting here for two hours; he has slumped, or shifted, or someone
     * moved his chair. The MAP from eye to screen is still right — it is his whole head that has
     * moved, which shows up as a constant offset. Three seconds looking at one dot fixes that,
     * where a full recalibration would cost him a minute he does not want to spend.
     */
    async recenter(nx, ny) {
      if (!model) return false;
      const want = { x: nx * window.innerWidth, y: ny * window.innerHeight };
      const got = [];
      const until = performance.now() + 2200;
      // The old bias stays in place until a new one is measured. Zeroing it first meant a failed
      // recenter (a blink, a look away) threw away all the drift correction learned so far.
      // Only NEW open-eye frames count (gazeSeq), so a turned-away head fails instead of locking
      // on to one stale frame. 15 frames still lets a ~10 fps CPU camera pass in 2.2 s.
      let seen = lastSample?.seq ?? -1;
      while (performance.now() < until) {
        await new Promise((r) => setTimeout(r, 40));
        if (!running || recovering) return false;
        if (!lastSample || !lastF || lastSample.seq === seen || lastLid > BLINK_ON || gate.gated) continue;
        seen = lastSample.seq;
        got.push(model.predict(lastF));
      }
      if (got.length < 15) return false;
      const mid = (k) => got.map((g) => g[k]).sort((a, b) => a - b)[Math.floor(got.length / 2)];
      // got is in calibration-time page pixels; the dot is where the page is now.
      const sh = pageShift();
      bias = { x: want.x - mid(0) - sh.x, y: want.y - mid(1) - sh.y };
      return { dx: Math.round(bias.x), dy: Math.round(bias.y) };
    },

    /** The unmapped eye signal itself, { v, seq }, or null (no face, or a blink). The signal check reads this. */
    raw() { return lastRaw; },

    /** The raw signal, for diagnosis. If gaze is wrong, the answer is in here. */
    probe() {
      return { backend, calibrated: !!model, running,
        features: 'head-fixed iris + head pose + lid shape (linear ridge)', camera, irisPx,
        errX: fit?.errX ?? null, errY: fit?.errY ?? null,
        lag: fit?.lag ?? null, lambda: fit?.lambda ?? null, terrain: terrain.length,
        noise: featureNoise() };
    },

    /** Where the gaze lands right now, unsmoothed — used by the accuracy test. */
    sample() {
      return lastSample;
    },

    stop() { teardown(); },
  };
}
