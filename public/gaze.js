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

const CAL_POINTS = [
  [0.5, 0.5],                                    // centre first: that is the neutral pose
  [0.08, 0.10], [0.5, 0.08], [0.92, 0.10],
  [0.06, 0.5],                [0.94, 0.5],
  [0.08, 0.90], [0.5, 0.92], [0.92, 0.90],
];
const SAMPLES_PER_POINT = 22;      // ~0.7s of frames
const SETTLE_MS = 600;             // let the eye actually land before we believe it
const BLINK_ON = 0.55;             // blendshape value above which a lid counts as closed
const BLINK_MIN_MS = 90;           // shorter than this is a twitch, not a blink
const BLINK_MAX_MS = 700;          // longer than this is a rest, not a blink

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

  // How far across the eye is the iris? 0 = against one corner, 1 = against the other.
  // Divided by the eye's own width, so it survives him leaning toward or away from the camera.
  const across = (iris, inner, outer) => {
    const w = lm[outer].x - lm[inner].x;
    return Math.abs(w) < 1e-6 ? 0.5 : (lm[iris].x - lm[inner].x) / w;
  };
  const between = (iris, top, bot) => {
    const h = lm[bot].y - lm[top].y;
    return Math.abs(h) < 1e-6 ? 0.5 : (lm[iris].y - lm[top].y) / h;
  };

  const ix = (across(IRIS_L, L_IN, L_OUT) + across(IRIS_R, R_IN, R_OUT)) / 2 - 0.5;
  const iy = (between(IRIS_L, L_TOP, L_BOT) + between(IRIS_R, R_TOP, R_BOT)) / 2 - 0.5;

  // Head yaw/pitch out of the 4x4 rigid transform (column-major).
  let yaw = 0, pitch = 0;
  if (matrix) {
    const m = matrix.data;
    yaw = Math.atan2(-m[8], Math.hypot(m[9], m[10]));
    pitch = Math.atan2(m[9], m[10]);
  }

  const lid = Math.max(b.eyeBlinkLeft ?? 0, b.eyeBlinkRight ?? 0);
  const bothShut = (b.eyeBlinkLeft ?? 0) > BLINK_ON && (b.eyeBlinkRight ?? 0) > BLINK_ON;

  // Cross terms let the fit correct gaze for head turn — looking left with the head turned left
  // is not the same eye position as looking left with the head straight.
  return {
    v: [ix, iy, yaw, pitch, ix * yaw, iy * pitch, 1],
    lid, bothShut,
  };
}

/**
 * Least squares: find the 5-vector w minimising |Xw - t|, via the normal equations.
 * X is [samples x 5] of gaze features, t is the screen coordinate we asked them to look at.
 * Ridge term keeps it from blowing up when someone barely moves their eyes during calibration.
 */
function solve(X, t, ridge = 1e-4) {
  const n = X[0].length;
  const A = Array.from({ length: n }, () => new Float64Array(n));
  const b = new Float64Array(n);
  for (let r = 0; r < X.length; r++) {
    for (let i = 0; i < n; i++) {
      b[i] += X[r][i] * t[r];
      for (let j = 0; j < n; j++) A[i][j] += X[r][i] * X[r][j];
    }
  }
  for (let i = 0; i < n; i++) A[i][i] += ridge;

  // Gaussian elimination with partial pivoting.
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    [b[c], b[p]] = [b[p], b[c]];
    if (Math.abs(A[c][c]) < 1e-12) continue;
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const w = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * w[j];
    w[i] = Math.abs(A[i][i]) < 1e-12 ? 0 : s / A[i][i];
  }
  return w;
}

const dot = (w, v) => w.reduce((s, wi, i) => s + wi * v[i], 0);

/**
 * THE OUTPUT STAGE. This is where "it just moves forever" was coming from.
 *
 * I was treating gaze like a mouse cursor: take the model's estimate every frame and glide the
 * dot toward it. But an eye does not glide. It JUMPS and then HOLDS (saccade, then fixation).
 * Chasing a per-frame estimate produces a dot that drifts forever and never settles on anything
 * — which is exactly what it did.
 *
 * So: reject the outliers, detect when the eye has actually LANDED, and freeze while it holds.
 */

/** Median of the last N — kills the single-frame spikes a mean would smear across the screen. */
function makeMedian(n = 7) {
  const bx = [], by = [];
  const mid = (a) => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)];
  return (x, y) => {
    bx.push(x); by.push(y);
    if (bx.length > n) { bx.shift(); by.shift(); }
    return [mid(bx), mid(by)];
  };
}

/**
 * Fixation detector. While the eye is moving, follow it fast. The moment it settles, LOCK —
 * and keep the dot dead still until it genuinely moves again.
 *
 * The lock is what makes the thing usable: a target that trembles under your gaze can never be
 * dwelled on, because every tremor resets the dwell.
 */
function makeFixation({ moveThresh = 55, holdThresh = 32, settleMs = 120 } = {}) {
  let px = null, py = null;        // reported position
  let lx = 0, ly = 0;              // last raw
  let stillSince = 0, locked = false;

  return (x, y, now) => {
    if (px === null) { px = x; py = y; lx = x; ly = y; stillSince = now; return [px, py, false]; }

    const step = Math.hypot(x - lx, y - ly);
    lx = x; ly = y;

    if (locked) {
      // Only break the lock on a real, sustained move — not on jitter.
      if (Math.hypot(x - px, y - py) > moveThresh) { locked = false; stillSince = now; }
      else return [px, py, true];
    }

    // Not locked: track, but heavily damped so it doesn't skate.
    px += 0.35 * (x - px);
    py += 0.35 * (y - py);

    if (step < holdThresh) {
      if (now - stillSince > settleMs) { locked = true; px = x; py = y; }
    } else {
      stillSince = now;
    }
    return [px, py, locked];
  };
}

export function createGaze({ onGaze, onBlink, onFace, onError, onCalibrationProgress }) {
  let landmarker = null, video = null, stream = null, backend = '?';
  let running = false, calibrating = false;
  let model = null;                       // { wx, wy } once calibrated
  let median = makeMedian();
  let fixate = makeFixation();
  let lastTs = -1;

  let blinkStart = 0, lidWasShut = false;
  let lastSample = null;      // the latest gaze point + raw features, for the accuracy test
  let calBucket = [], calResolve = null;

  function teardown() {
    running = false;
    stream?.getTracks().forEach((t) => t.stop());
    video?.remove();
    video = stream = null;
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

    if (video.readyState < 2 || video.currentTime === lastTs) return;
    lastTs = video.currentTime;

    let out;
    try { out = landmarker.detectForVideo(video, performance.now()); }
    catch { return; }

    const face = out.faceBlendshapes?.[0]?.categories;
    const lm = out.faceLandmarks?.[0];
    // 478 landmarks means the iris points are there. 468 means they are not, and gaze is dead.
    if (!face?.length || !lm || lm.length < 478) { onFace(false); return; }
    onFace(true);

    const f = features(lm, face, out.facialTransformationMatrixes?.[0]);

    // A deliberate blink is a SELECT. Reflex blinks are ~100-150ms and constant; we require the
    // lids to stay shut a beat longer than that, but not so long it's just a rest.
    if (f.bothShut && !lidWasShut) { lidWasShut = true; blinkStart = performance.now(); }
    else if (!f.bothShut && lidWasShut) {
      lidWasShut = false;
      const held = performance.now() - blinkStart;
      if (held > BLINK_MIN_MS && held < BLINK_MAX_MS && !calibrating) onBlink(held);
    }

    if (calibrating) { calBucket.push(f.v); return; }
    if (!model) return;

    // Eyes shut: hold the last position. Otherwise the cursor lurches away every blink.
    if (f.lid > BLINK_ON) return;

    // A LINEAR fit extrapolates without limit: glance past the edge of the calibrated range and
    // it happily reports a point three screens away, dragging the dot off into space. Clamp it.
    const rx = Math.max(0, Math.min(window.innerWidth, dot(model.wx, f.v)));
    const ry = Math.max(0, Math.min(window.innerHeight, dot(model.wy, f.v)));

    const now = performance.now();
    const [mx, my] = median(rx, ry);
    const [x, y, locked] = fixate(mx, my, now);

    lastSample = { x, y, locked, raw: f.v.map((n) => +n.toFixed(3)) };
    onGaze(x, y, locked);
  }

  return {
    get calibrated() { return !!model; },
    get running() { return running; },
    get backend() { return backend; },

    async start() {
      // getUserMedia can hang forever — no camera, or one another app is holding. A promise that
      // never settles means no error, no toast, and a user tapping a button that says nothing
      // back. Every await here gets a deadline.
      const deadline = (p, ms, what) => Promise.race([
        p,
        new Promise((_, rej) => setTimeout(() => rej(new Error(`${what} timed out`)), ms)),
      ]);

      try {
        stream = await deadline(
          navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } }),
          12000, 'camera',
        );
      } catch (e) {
        onError(e.name === 'NotAllowedError'
          ? 'Camera blocked. Allow it in the address bar, then turn Eye tracking back on.'
          : e.name === 'NotFoundError' ? 'No camera found on this machine.'
          : `Camera: ${e.message}`);
        return false;
      }

      video = document.createElement('video');
      video.autoplay = true; video.playsInline = true; video.muted = true;
      video.srcObject = stream;
      try { await deadline(video.play(), 5000, 'video'); }
      catch (e) { onError(`Camera opened but no frames: ${e.message}`); teardown(); return false; }

      try { if (!landmarker) await deadline(load(), 25000, 'face model'); }
      catch (e) { onError(`Eye tracking unavailable — ${e.message}`); teardown(); return false; }

      running = true;
      loop();
      return true;
    },

    /** Look at each dot. We record where the eyes go, then fit the map from eye → screen. */
    async calibrate() {
      if (!running) return false;
      const X = [], tx = [], ty = [];

      for (let i = 0; i < CAL_POINTS.length; i++) {
        const [px, py] = CAL_POINTS[i];
        onCalibrationProgress({ index: i, total: CAL_POINTS.length, x: px, y: py, state: 'settle' });
        await new Promise((r) => setTimeout(r, SETTLE_MS));

        calBucket = [];
        calibrating = true;
        onCalibrationProgress({ index: i, total: CAL_POINTS.length, x: px, y: py, state: 'sampling' });

        await new Promise((resolve) => {
          const check = setInterval(() => {
            if (calBucket.length >= SAMPLES_PER_POINT) { clearInterval(check); resolve(); }
          }, 30);
          setTimeout(() => { clearInterval(check); resolve(); }, 3000);  // don't hang on a lost face
        });
        calibrating = false;

        // Drop the first few: the eye is often still travelling when sampling opens.
        const keep = calBucket.slice(Math.floor(calBucket.length * 0.3));
        for (const v of keep) {
          X.push(v);
          tx.push(px * window.innerWidth);
          ty.push(py * window.innerHeight);
        }
      }

      if (X.length < 40) { onError('Calibration failed — your face was not visible enough.'); return false; }

      // FIT, THEN THROW AWAY THE WORST, THEN FIT AGAIN.
      // Least squares has no defence against a bad sample: one frame where he blinked, or where
      // the eye was still travelling to the dot, tilts the entire plane. Two refits, each
      // dropping the worst 20% of residuals, and the fit stops being hostage to a few frames.
      let wx = solve(X, tx), wy = solve(X, ty);
      for (let pass = 0; pass < 2; pass++) {
        const res = X.map((v, i) => Math.hypot(dot(wx, v) - tx[i], dot(wy, v) - ty[i]));
        const cut = [...res].sort((a, b) => a - b)[Math.floor(res.length * 0.8)];
        const kx = [], ktx = [], kty = [];
        for (let i = 0; i < X.length; i++) {
          if (res[i] <= cut) { kx.push(X[i]); ktx.push(tx[i]); kty.push(ty[i]); }
        }
        if (kx.length < 30) break;
        wx = solve(kx, ktx); wy = solve(kx, kty);
      }

      model = { wx, wy };
      median = makeMedian();
      fixate = makeFixation();

      // Honest self-check: how far off is the fit on its own training points? If it can't even
      // reproduce those, the tracking will be useless and the user deserves to know now.
      let err = 0;
      const worst = [];
      for (let i = 0; i < X.length; i++) {
        const e = Math.hypot(dot(model.wx, X[i]) - tx[i], dot(model.wy, X[i]) - ty[i]);
        err += e;
        worst.push(e);
      }
      worst.sort((a, b) => a - b);
      const px = err / X.length;
      onCalibrationProgress({
        state: 'done',
        errorPx: Math.round(px),
        p90Px: Math.round(worst[Math.floor(worst.length * 0.9)] ?? px),
        samples: X.length,
        backend,
      });
      return true;
    },

    recalibrate() { model = null; },

    /** The raw signal, for diagnosis. If gaze is wrong, the answer is in here. */
    probe() {
      return { backend, calibrated: !!model, running, features: 'iris+head',
        model: model ? {
          wx: [...model.wx].map((n) => +n.toFixed(1)),
          wy: [...model.wy].map((n) => +n.toFixed(1)),
        } : null };
    },

    /** Where the gaze lands right now, unsmoothed — used by the accuracy test. */
    sample() {
      return lastSample;
    },

    stop() { teardown(); },
  };
}
