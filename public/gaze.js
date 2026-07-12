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

/**
 * Where the eyes are pointing, plus where the head is pointing.
 *
 * People aim at things with BOTH. Using eye blendshapes alone drifts badly the moment someone
 * turns their head; using head pose alone can't tell a glance from a stare. Feeding both to the
 * calibration fit lets it work out the mix for this particular person, in this particular chair.
 */
function features(face, matrix) {
  const b = {};
  for (const c of face) b[c.categoryName] = c.score;

  // Anatomy: looking right ADDUCTS the left eye (in, toward the nose) and ABDUCTS the right eye.
  const x = (b.eyeLookInLeft ?? 0) + (b.eyeLookOutRight ?? 0)
          - (b.eyeLookOutLeft ?? 0) - (b.eyeLookInRight ?? 0);
  const y = (b.eyeLookUpLeft ?? 0) + (b.eyeLookUpRight ?? 0)
          - (b.eyeLookDownLeft ?? 0) - (b.eyeLookDownRight ?? 0);

  // Head yaw/pitch out of the 4x4 rigid transform (column-major).
  let yaw = 0, pitch = 0;
  if (matrix) {
    const m = matrix.data;
    yaw = Math.atan2(-m[8], Math.hypot(m[9], m[10]));
    pitch = Math.atan2(m[9], m[10]);
  }

  const lid = Math.max(b.eyeBlinkLeft ?? 0, b.eyeBlinkRight ?? 0);
  const bothShut = (b.eyeBlinkLeft ?? 0) > BLINK_ON && (b.eyeBlinkRight ?? 0) > BLINK_ON;

  return { v: [x, y, yaw, pitch, 1], lid, bothShut };
}

/**
 * Least squares: find the 5-vector w minimising |Xw - t|, via the normal equations.
 * X is [samples x 5] of gaze features, t is the screen coordinate we asked them to look at.
 * Ridge term keeps it from blowing up when someone barely moves their eyes during calibration.
 */
function solve(X, t, ridge = 1e-4) {
  const n = 5;
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
 * One-Euro-ish smoothing: heavy when the eye is resting (kills jitter, so the cursor doesn't
 * shiver off the tile he's trying to dwell on), light when it's moving (so it doesn't lag).
 * A fixed low-pass would force a choice between a shaky cursor and a sluggish one.
 */
function makeSmoother(minAlpha = 0.08, maxAlpha = 0.9, speedScale = 900) {
  let px = null, py = null;
  return (x, y) => {
    if (px === null) { px = x; py = y; return [x, y]; }
    const speed = Math.hypot(x - px, y - py);
    const a = Math.min(maxAlpha, minAlpha + (speed / speedScale) * (maxAlpha - minAlpha));
    px += a * (x - px);
    py += a * (y - py);
    return [px, py];
  };
}

export function createGaze({ onGaze, onBlink, onFace, onError, onCalibrationProgress }) {
  let landmarker = null, video = null, stream = null, backend = '?';
  let running = false, calibrating = false;
  let model = null;                       // { wx, wy } once calibrated
  let smooth = makeSmoother();
  let lastTs = -1;

  let blinkStart = 0, lidWasShut = false;
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
    if (!face || !face.length) { onFace(false); return; }
    onFace(true);

    const f = features(face, out.facialTransformationMatrixes?.[0]);

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

    const [x, y] = smooth(dot(model.wx, f.v), dot(model.wy, f.v));
    onGaze(x, y);
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

      model = { wx: solve(X, tx), wy: solve(X, ty) };
      smooth = makeSmoother();

      // Honest self-check: how far off is the fit on its own training points? If it can't even
      // reproduce those, the tracking will be useless and the user deserves to know now.
      let err = 0;
      for (let i = 0; i < X.length; i++) {
        err += Math.hypot(dot(model.wx, X[i]) - tx[i], dot(model.wy, X[i]) - ty[i]);
      }
      const px = err / X.length;
      onCalibrationProgress({ state: 'done', errorPx: Math.round(px) });
      return true;
    },

    recalibrate() { model = null; },

    stop() { teardown(); },
  };
}
