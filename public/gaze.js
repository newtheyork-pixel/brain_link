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
 * WHY THE DOT FLEW AROUND WHILE HIS EYES WERE STILL.
 *
 * I fitted a least-squares LINE from eye-features to screen pixels. With seven near-collinear
 * features the normal equations are badly conditioned, and the solver answers with ENORMOUS
 * weights — thousands of pixels per unit of feature. The iris landmark wobbles by a fraction of
 * a percent between frames (it always does), that wobble gets multiplied by a huge weight, and
 * the dot leaps across the screen while the man sits perfectly still.
 *
 * I built an amplifier and called it a tracker.
 *
 * So: no regression. INTERPOLATE INSTEAD.
 *
 * Calibration stores, for each of the nine dots, the average eye-feature vector while he looked
 * at it. At runtime we ask: which of those nine does the eye look most like right now? The
 * answer is a weighted blend of the nine screen positions — so the output is ALWAYS inside the
 * region he calibrated. It cannot extrapolate. It cannot amplify. A small wobble in the iris
 * makes a small wobble in the blend, and nothing else.
 *
 * (Shepard / inverse-distance interpolation, in a feature space standardised so that a
 * millimetre of iris and a degree of head-turn count the same.)
 */
function buildModel(points) {
  const dim = points[0].mean.length;

  // Standardise: without this, head-yaw (radians, ~0.3) would drown iris position (~0.05), and
  // the "nearest" calibration point would be decided almost entirely by how he held his head.
  const mu = new Float64Array(dim);
  const sd = new Float64Array(dim);
  for (const p of points) for (let i = 0; i < dim; i++) mu[i] += p.mean[i] / points.length;
  for (const p of points) {
    for (let i = 0; i < dim; i++) sd[i] += (p.mean[i] - mu[i]) ** 2 / points.length;
  }
  for (let i = 0; i < dim; i++) sd[i] = Math.sqrt(sd[i]) || 1e-3;

  const z = (v) => v.map((x, i) => (x - mu[i]) / sd[i]);
  const centroids = points.map((p) => ({ z: z(p.mean), x: p.x, y: p.y }));

  return {
    dim,
    predict(v) {
      const q = z(v);
      let wsum = 0, sx = 0, sy = 0;
      for (const c of centroids) {
        let d2 = 0;
        for (let i = 0; i < dim; i++) d2 += (q[i] - c.z[i]) ** 2;
        // Sharp enough to pick a corner, soft enough to slide smoothly between neighbours.
        const w = 1 / (d2 * d2 + 0.02);
        wsum += w; sx += w * c.x; sy += w * c.y;
      }
      return [sx / wsum, sy / wsum];   // ALWAYS a blend of calibrated points. Bounded, always.
    },
    // How separated were the nine calibration points in feature space? If the eye barely moved
    // between them, no algorithm on earth can tell them apart, and he deserves to be told.
    spread() {
      let min = Infinity;
      for (let i = 0; i < centroids.length; i++) {
        for (let j = i + 1; j < centroids.length; j++) {
          let d2 = 0;
          for (let k = 0; k < dim; k++) d2 += (centroids[i].z[k] - centroids[j].z[k]) ** 2;
          min = Math.min(min, Math.sqrt(d2));
        }
      }
      return min;
    },
  };
}

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
  let lastSample = null;      // the latest gaze point + raw features, for the accuracy test
  let lastRaw = null;         // the unsmoothed feature vector — the signal check reads this
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

    const fRaw = features(lm, face, out.facialTransformationMatrixes?.[0]);
    lastRaw = fRaw.v;
    trackNoise(fRaw.v);
    const f = { ...fRaw, v: smoothFeatures(fRaw.v) };

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

    const [rx, ry] = model.predict(f.v);   // bounded by construction — always inside the fit

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

    /** Look at each dot. We record what his eyes LOOK LIKE at each, then interpolate between. */
    async calibrate() {
      if (!running) return false;
      const points = [];

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
          setTimeout(() => { clearInterval(check); resolve(); }, 3500);   // never hang on a lost face
        });
        calibrating = false;

        // Drop the first third — the eye is often still travelling when sampling opens.
        const keep = calBucket.slice(Math.floor(calBucket.length * 0.35));
        if (keep.length < 6) { onError('Calibration failed — your face was not visible enough.'); return false; }

        // The MEDIAN of each feature, not the mean: one blink during this dot must not drag the
        // whole point sideways.
        const dim = keep[0].length;
        const mean = [];
        for (let k = 0; k < dim; k++) {
          const col = keep.map((v) => v[k]).sort((a, b) => a - b);
          mean.push(col[Math.floor(col.length / 2)]);
        }
        points.push({ mean, x: px * window.innerWidth, y: py * window.innerHeight, n: keep.length });
      }

      model = buildModel(points);
      median = makeMedian();
      fixate = makeFixation();

      // How far apart were the nine points in feature space? If his eyes barely moved between
      // the corners of the screen — sitting too far back, or a camera that cannot resolve the
      // iris — then nothing downstream can separate them, and he needs to know THAT, not be
      // handed a cursor that lies.
      const spread = model.spread();

      // Honest residual: re-predict each calibration point from its own features.
      let err = 0;
      for (const pt of points) {
        const [x, y] = model.predict(pt.mean);
        err += Math.hypot(x - pt.x, y - pt.y);
      }
      const errorPx = Math.round(err / points.length);

      onCalibrationProgress({
        state: 'done', errorPx, spread: +spread.toFixed(2), backend,
        weak: spread < 0.8,
        samples: points.reduce((a, p) => a + p.n, 0),
      });
      return true;
    },

    recalibrate() { model = null; },

    /** The unmapped eye signal itself. The signal check needs this, not the screen point. */
    raw() { return lastRaw; },

    /** The raw signal, for diagnosis. If gaze is wrong, the answer is in here. */
    probe() {
      return { backend, calibrated: !!model, running, features: 'iris+head',
        method: 'inverse-distance interpolation (bounded)',
        spread: model ? +model.spread().toFixed(2) : null,
        noise: featureNoise() };
    },

    /** Where the gaze lands right now, unsmoothed — used by the accuracy test. */
    sample() {
      return lastSample;
    },

    stop() { teardown(); },
  };
}
