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

// CALIBRATE WHERE THE TILES ARE, not at the corners of the glass.
//
// I was calibrating at the extreme top and bottom edges of the screen. The two worst points in
// the last run were both at the very bottom — because extreme downgaze is exactly where the upper
// eyelid comes down over the iris and the signal falls apart. Those points were teaching the model
// nonsense, and it then applied that nonsense everywhere.
//
// He never looks at the bottom edge of the glass. He looks at TILES. So calibrate there: the app
// hands us the real tile centres, and we add a modest margin around them so the fit is
// interpolating across his working area rather than extrapolating out of it.
const DEFAULT_CAL = [
  [0.5, 0.5],
  [0.12, 0.18], [0.5, 0.16], [0.88, 0.18],
  [0.10, 0.5],                [0.90, 0.5],
  [0.12, 0.82], [0.5, 0.84], [0.88, 0.82],
];
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

  // THE TWO EYES WERE CANCELLING EACH OTHER OUT.
  //
  // I measured each iris as a fraction from the INNER corner toward the OUTER corner. But "outer"
  // is in opposite image directions for the two eyes: for one eye the outer corner has a SMALLER
  // x than the inner, for the other a LARGER one. So the two denominators carried opposite signs.
  // When he looked right, both irises moved right in the image — and the two normalised readings
  // moved in OPPOSITE directions and annihilated each other in the average.
  //
  // The measured proof: looking from the far left of the screen to the far right moved the signal
  // by 0.0075. It should move by ~0.2. Thirty times too small, because it was mostly cancelling.
  //
  // Fix: measure each iris as an offset from ITS OWN EYE'S CENTRE, in image coordinates. Same
  // sign for both eyes, always.
  const irisCentre = (a, z) => {
    let x = 0, y = 0;
    for (let i = a; i <= z; i++) { x += lm[i].x; y += lm[i].y; }
    return { x: x / (z - a + 1), y: y / (z - a + 1) };   // the ring, not one point: 5x less jitter
  };

  const eye = (irisA, irisZ, c1, c2, top, bot) => {
    const ir = irisCentre(irisA, irisZ);
    const cx = (lm[c1].x + lm[c2].x) / 2;
    const cy = (lm[c1].y + lm[c2].y) / 2;
    const w = Math.hypot(lm[c2].x - lm[c1].x, lm[c2].y - lm[c1].y) || 1e-6;

    // Vertical is normalised by eye WIDTH, not eye height. The lid-to-lid distance is tiny and
    // it moves every time he blinks or squints — dividing by it produced a vertical signal with
    // a noise reading of 1.7 (i.e. pure garbage). Eye width is rigid and stable.
    return { x: (ir.x - cx) / w, y: (ir.y - cy) / w };
  };

  const L = eye(468, 472, L_OUT, L_IN, L_TOP, L_BOT);
  const R = eye(473, 477, R_IN, R_OUT, R_TOP, R_BOT);

  const ix = (L.x + R.x) / 2;
  const iy = (L.y + R.y) / 2;

  // Lid opening, normalised by eye width. Looking DOWN closes the lid; looking UP opens it. It is
  // one of the strongest vertical-gaze cues on a webcam — and without it the model cannot separate
  // a downward eye from a drooping lid, which is exactly what wrecked the bottom of the screen.
  const aperture = (top, bot, c1, c2) => {
    const h = Math.hypot(lm[bot].x - lm[top].x, lm[bot].y - lm[top].y);
    const w = Math.hypot(lm[c2].x - lm[c1].x, lm[c2].y - lm[c1].y) || 1e-6;
    return h / w;
  };
  const ap = (aperture(L_TOP, L_BOT, L_OUT, L_IN) + aperture(R_TOP, R_BOT, R_IN, R_OUT)) / 2;

  // Head yaw/pitch out of the 4x4 rigid transform (column-major).
  let yaw = 0, pitch = 0;
  if (matrix) {
    const m = matrix.data;
    yaw = Math.atan2(-m[8], Math.hypot(m[9], m[10]));
    pitch = Math.atan2(m[9], m[10]);
  }

  const lid = Math.max(b.eyeBlinkLeft ?? 0, b.eyeBlinkRight ?? 0);
  const bothShut = (b.eyeBlinkLeft ?? 0) > BLINK_ON && (b.eyeBlinkRight ?? 0) > BLINK_ON;

  // No cross-terms. They multiply two noisy numbers together and hand the result to a model that
  // has to tell an 8th of a screen apart. Iris and head pose, and nothing else.
  return { v: [ix, iy, yaw, pitch, ap], lid, bothShut };
}

/**
 * THE MODEL. Third attempt, and this time the error it reports is a real one.
 *
 * Attempt 1 — least squares on 7 collinear features, no standardisation. Produced weights of
 * TWENTY THOUSAND pixels per unit. A 0.3% iris wobble became 60px of jitter. An amplifier.
 *
 * Attempt 2 — inverse-distance interpolation between 9 calibration centroids. Bounded, so it
 * stopped flying off screen. But it reported a 5-pixel calibration error, which was a LIE: the
 * weight at zero distance is enormous, so each point trivially reproduced ITSELF. It was
 * grading its own homework with the answers in front of it. And because every feature was
 * standardised to equal weight, HEAD POSE counted as much as iris position — so "where is he
 * looking" was decided partly by how he was holding his head.
 *
 * Attempt 3 — what actually works, and what real webcam trackers do:
 *
 *   - IRIS ONLY drives the mapping. He sits at a laptop; his head is roughly fixed. Letting head
 *     pose into the model just lets a shrug hijack the estimate.
 *   - A quadratic surface (1, x, y, x², y², xy). Gaze-to-screen is mildly curved; a plane can't
 *     fit the corners.
 *   - Features STANDARDISED before solving, and a ridge scaled to the data. This is exactly what
 *     was missing in attempt 1 — it is why the weights exploded.
 *   - The reported error is LEAVE-ONE-OUT: fit on 8 points, predict the 9th, and measure that.
 *     A model cannot cheat on a point it has never seen. THIS is the number that tells the truth.
 */

const poly = (ix, iy) => [1, ix, iy, ix * ix, iy * iy, ix * iy];

function ridgeSolve(X, t, lambda) {
  const n = X[0].length;
  const A = Array.from({ length: n }, () => new Float64Array(n));
  const b = new Float64Array(n);
  for (let r = 0; r < X.length; r++) {
    for (let i = 0; i < n; i++) {
      b[i] += X[r][i] * t[r];
      for (let j = 0; j < n; j++) A[i][j] += X[r][i] * X[r][j];
    }
  }
  for (let i = 1; i < n; i++) A[i][i] += lambda;   // never penalise the intercept

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

const dotv = (w, v) => w.reduce((s, wi, i) => s + wi * v[i], 0);

/**
 * I dropped head pose on a hunch and the error got WORSE. So stop hunching.
 *
 * When a man looks at the far corner of a laptop screen he TURNS HIS HEAD. His eyes do only part
 * of the work — which means the iris alone cannot tell you where he is looking, because the same
 * iris position means different things depending on where the head is pointing. That is why an
 * iris-only fit could not predict a held-out corner.
 *
 * So fit BOTH candidates and let leave-one-out cross-validation pick the winner. The data decides
 * which features matter, not me. I have now guessed wrong three times; the guessing stops here.
 */
/**
 * THE MODEL, now that we have real data.
 *
 * Nine points could only ever support a stiff quadratic. With HUNDREDS of samples from a smooth
 * pursuit we can fit a surface that actually bends — which matters, because the map from eye to
 * screen is genuinely curved, and a stiff surface splits the difference by being wrong everywhere.
 *
 * Radial basis functions: lay Gaussian bumps across the space of eye positions, and let the fit
 * decide how much of each. Locally flexible, globally smooth, and — with ridge and honest
 * validation — it cannot run away like the linear fit did.
 *
 * X and Y stay separate models. Horizontal is nearly solved by the iris alone; vertical needs the
 * lid, because a lowered eyelid and a lowered eye look the same to a camera.
 */
const N_RBF = 5;   // 5x5 = 25 Gaussian centres across the eye's range

function makeBasis(samples, extract) {
  const raw = samples.map(extract);
  const dim = raw[0].length;

  const mu = [], sg = [];
  for (let k = 0; k < dim; k++) {
    const col = raw.map((v) => v[k]);
    const m = col.reduce((s, v) => s + v, 0) / col.length;
    mu.push(m);
    sg.push(Math.sqrt(col.reduce((s, v) => s + (v - m) ** 2, 0) / col.length) || 1e-4);
  }
  const z = (v) => v.map((x, k) => (x - mu[k]) / sg[k]);

  // Centres on a grid over the first two standardised dimensions — the ones that carry gaze.
  const centres = [];
  const lo = -1.8, hi = 1.8, stepC = (hi - lo) / (N_RBF - 1);
  for (let i = 0; i < N_RBF; i++) {
    for (let j = 0; j < N_RBF; j++) centres.push([lo + i * stepC, lo + j * stepC]);
  }
  const gamma = 1 / (2 * stepC * stepC);   // bumps overlap their neighbours; no bald patches

  return {
    dim,
    feat(v) {
      const q = z(v);
      const f = [1, ...q];                                  // linear trend (lid, head ride here)
      for (const c of centres) {
        const d2 = (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2;
        f.push(Math.exp(-gamma * d2));
      }
      return f;
    },
  };
}

/** Fit one axis on `train`, and report the error on data it has NEVER SEEN. */
function fitAxis(train, test, targetKey, variants) {
  let best = null;
  const scores = {};

  for (const [name, extract] of Object.entries(variants)) {
    const B = makeBasis(train, extract);
    const X = train.map((s) => B.feat(extract(s)));
    const t = train.map((s) => s[targetKey]);

    for (const lambda of [0.3, 1, 3, 10, 30, 100]) {
      const w = ridgeSolve(X, t, lambda);
      // Honest: a DIFFERENT pass of the calibration, collected at a different time. Not a
      // random split of the same frames, which would leak — neighbouring frames are near-copies.
      let err = 0;
      for (const s of test) err += Math.abs(dotv(w, B.feat(extract(s))) - s[targetKey]);
      err /= test.length;
      const key = `${name}`;
      if (scores[key] === undefined || err < scores[key]) scores[key] = Math.round(err);
      if (!best || err < best.err) best = { err, lambda, name, extract, B };
    }
  }

  // Refit the winner on EVERYTHING — the held-out pass was for choosing, and now it is data.
  const all = [...train, ...test];
  const Ball = makeBasis(all, best.extract);
  const w = ridgeSolve(all.map((s) => Ball.feat(best.extract(s))), all.map((s) => s[targetKey]), best.lambda);

  return {
    scores, name: best.name, err: Math.round(best.err), lambda: best.lambda,
    predict: (p) => dotv(w, Ball.feat(best.extract(p))),
    weightMax: Math.round(Math.max(...[...w].map(Math.abs))),
  };
}

// A person shifts in their chair. They lean. They slump. If the model has only ever seen one head
// position it has no idea what to do with a different one — and the runtime data showed exactly
// that: a systematic +0.023 offset in iris-x, ~18% of his whole gaze range, purely from his head
// having moved between calibrating and using it.
const X_VARIANTS = {
  iris: (p) => [p.ix, p.iy],
  'iris+yaw': (p) => [p.ix, p.iy, p.yaw],
  'iris+head': (p) => [p.ix, p.iy, p.yaw, p.pitch],
};
const Y_VARIANTS = {
  iris: (p) => [p.iy, p.ix],
  'iris+lid': (p) => [p.iy, p.ix, p.ap],
  'iris+head': (p) => [p.iy, p.ix, p.pitch, p.yaw],
  'iris+lid+head': (p) => [p.iy, p.ix, p.ap, p.pitch, p.yaw],
};

function buildModel(train, test) {
  const X = fitAxis(train, test, 'x', X_VARIANTS);
  const Y = fitAxis(train, test, 'y', Y_VARIANTS);
  return {
    errX: X.err, errY: Y.err,
    looErrorPx: Math.round(Math.hypot(X.err, Y.err)),
    variant: `x:${X.name} y:${Y.name}`,
    looByVariant: { x: X.scores, y: Y.scores },
    lambda: `${X.lambda}/${Y.lambda}`,
    weightMax: Math.max(X.weightMax, Y.weightMax),
    nTrain: train.length, nTest: test.length,
    predict(p) {
      return [
        Math.max(0, Math.min(window.innerWidth, X.predict(p))),
        Math.max(0, Math.min(window.innerHeight, Y.predict(p))),
      ];
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
  let camera = null, irisPx = 0;
  let running = false;
  let model = null;                       // the eye→screen map, once calibrated
  let bias = { x: 0, y: 0 };              // constant drift correction, from recenter()
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
  let lastLid = 0;            // reject frames where he blinked: they carry no gaze at all

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

    // How many pixels wide is the iris, really? Below ~12 the landmark cannot resolve where it
    // is sitting, and no algorithm downstream can recover that.
    if (camera) {
      irisPx = Math.round(Math.hypot(lm[471].x - lm[469].x, lm[471].y - lm[469].y) * camera.w);
    }

    const fRaw = features(lm, face, out.facialTransformationMatrixes?.[0]);
    lastRaw = fRaw.v;
    lastLid = fRaw.lid;
    trackNoise(fRaw.v);
    const f = { ...fRaw, v: smoothFeatures(fRaw.v) };

    // A deliberate blink is a SELECT. Reflex blinks are ~100-150ms and constant; we require the
    // lids to stay shut a beat longer than that, but not so long it's just a rest.
    if (f.bothShut && !lidWasShut) { lidWasShut = true; blinkStart = performance.now(); }
    else if (!f.bothShut && lidWasShut) {
      lidWasShut = false;
      const held = performance.now() - blinkStart;
      if (held > BLINK_MIN_MS && held < BLINK_MAX_MS) onBlink(held);
    }

    if (!model) return;

    // Eyes shut: hold the last position. Otherwise the cursor lurches away every blink.
    if (f.lid > BLINK_ON) return;

    const [px0, py0] = model.predict({ ix: f.v[0], iy: f.v[1], yaw: f.v[2], pitch: f.v[3], ap: f.v[4] });
    const rx = Math.max(0, Math.min(window.innerWidth, px0 + bias.x));
    const ry = Math.max(0, Math.min(window.innerHeight, py0 + bias.y));

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
        // 640x480 leaves the iris about ten pixels across at laptop distance — the landmark then
        // quantises to that grid and the jitter IS the signal. Ask for everything the camera has.
        stream = await deadline(
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

      const t = stream.getVideoTracks()[0]?.getSettings?.() ?? {};
      camera = { w: t.width ?? 0, h: t.height ?? 0, fps: t.frameRate ?? 0 };

      running = true;
      loop();
      return true;
    },

    /** Look at each dot. We record what his eyes LOOK LIKE at each, then interpolate between. */
    /**
     * SMOOTH PURSUIT CALIBRATION.
     *
     * Nine dots gave nine data points and a stiff, wrong surface. But the eye is very good at
     * FOLLOWING a slowly moving target — that is a reflex, not a skill — and while it follows we
     * can sample continuously. One 40-second pass yields hundreds of (eye → screen) pairs instead
     * of nine, densely covering the whole working area rather than sampling it at the corners.
     *
     * He is going to live inside this device. Five minutes of calibration, once, is nothing set
     * against years of use — and it is the difference between a toy and a tool.
     *
     * TWO PASSES, deliberately. The first trains the model; the second is a held-out validation
     * collected at a DIFFERENT TIME, on a DIFFERENT PATH. That is an honest error. (Randomly
     * splitting one pass would leak: neighbouring frames are near-copies of each other.)
     */
    async calibrate({ bounds, onSample } = {}) {
      if (!running) return false;

      // Stay inside his working area. The far edges of the glass are where the eyelid swallows the
      // iris, and those samples taught the model nonsense which it then applied everywhere.
      const B = bounds ?? { x0: 0.10, x1: 0.90, y0: 0.16, y1: 0.86 };
      const lerp = (a, b, t) => a + (b - a) * t;

      // Serpentine, then the same area traversed the other way. Different paths mean the second
      // pass is a genuine test, not a rerun.
      const path = (pass) => {
        const pts = [];
        const ROWS = 5, STEPS = 26;
        for (let r = 0; r < ROWS; r++) {
          const t = r / (ROWS - 1);
          for (let i = 0; i < STEPS; i++) {
            const u = i / (STEPS - 1);
            const sweep = r % 2 ? 1 - u : u;
            pts.push(pass === 0
              ? [lerp(B.x0, B.x1, sweep), lerp(B.y0, B.y1, t)]     // across, then down
              : [lerp(B.x0, B.x1, t), lerp(B.y0, B.y1, sweep)]);   // down, then across
          }
        }
        return pts;
      };

      const collected = [[], []];
      const HOLD = 78;          // ms per waypoint — slow enough for the eye to actually keep up
      const SETTLE = 3;         // waypoints to discard after each turn, while the eye catches up

      for (let pass = 0; pass < 2; pass++) {
        const pts = path(pass);
        onCalibrationProgress({ state: 'pursuit', pass, total: 2, x: pts[0][0], y: pts[0][1],
          progress: 0, moveHead: pass === 1 });
        await new Promise((r) => setTimeout(r, pass === 1 ? 3000 : 1400));

        for (let i = 0; i < pts.length; i++) {
          const [px, py] = pts[i];
          onCalibrationProgress({ state: 'pursuit', pass, total: 2, x: px, y: py,
            progress: (i + 1) / pts.length });

          const until = performance.now() + HOLD;
          while (performance.now() < until) {
            await new Promise((r) => setTimeout(r, 16));
            if (!lastRaw || i < SETTLE) continue;
            if (lastLid > BLINK_ON) continue;            // he blinked; that frame is worthless
            collected[pass].push({
              ix: lastRaw[0], iy: lastRaw[1], yaw: lastRaw[2], pitch: lastRaw[3], ap: lastRaw[4],
              x: px * window.innerWidth, y: py * window.innerHeight,
            });
          }
          onSample?.(collected[0].length + collected[1].length);
        }
      }

      const [train, test] = collected;
      const all = [...train, ...test];

      // Did his head actually move? If not, the model has no way to learn the correction, and it
      // WILL break the moment he shifts in his chair — which is exactly what happened.
      const spread = (k) => {
        const v = all.map((p) => p[k]);
        const m = v.reduce((a, b) => a + b, 0) / v.length;
        return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
      };
      const headSpread = +Math.max(spread('yaw'), spread('pitch')).toFixed(4);

      if (train.length < 60 || test.length < 40) {
        onError('Calibration failed — your face was not visible enough. More light, sit closer.');
        return false;
      }

      model = buildModel(train, test);
      median = makeMedian();
      fixate = makeFixation();

      const t = document.querySelector('.tile')?.getBoundingClientRect();
      const tileW = t?.width || window.innerWidth / 4;
      const tileH = t?.height || window.innerHeight / 2;
      const usable = model.errX < tileW * 0.45 && model.errY < tileH * 0.45;

      onCalibrationProgress({
        state: 'done', errorPx: model.looErrorPx, errX: model.errX, errY: model.errY,
        usable, backend, variant: model.variant, looByVariant: model.looByVariant,
        lambda: model.lambda, weightMax: model.weightMax,
        samples: train.length + test.length, nTrain: model.nTrain, nTest: model.nTest,
        headSpread, headVaried: headSpread > 0.05,
        tile: { w: Math.round(tileW), h: Math.round(tileH) },
      });
      return true;
    },

    recalibrate() { model = null; bias = { x: 0, y: 0 }; },

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
      bias = { x: 0, y: 0 };
      while (performance.now() < until) {
        await new Promise((r) => setTimeout(r, 40));
        if (!lastRaw || lastLid > BLINK_ON) continue;
        got.push(model.predict({ ix: lastRaw[0], iy: lastRaw[1], yaw: lastRaw[2],
          pitch: lastRaw[3], ap: lastRaw[4] }));
      }
      if (got.length < 20) return false;
      const mid = (k) => got.map((g) => g[k]).sort((a, b) => a - b)[Math.floor(got.length / 2)];
      bias = { x: want.x - mid(0), y: want.y - mid(1) };
      return { dx: Math.round(bias.x), dy: Math.round(bias.y) };
    },

    /** The unmapped eye signal itself. The signal check needs this, not the screen point. */
    raw() { return lastRaw; },

    /** The raw signal, for diagnosis. If gaze is wrong, the answer is in here. */
    probe() {
      return { backend, calibrated: !!model, running, features: 'iris (quadratic ridge)',
        camera, irisPx,
        looErrorPx: model?.looErrorPx ?? null,
        lambda: model?.lambda ?? null,
        weightMax: model?.weightMax ?? null,
        noise: featureNoise() };
    },

    /** Where the gaze lands right now, unsmoothed — used by the accuracy test. */
    sample() {
      return lastSample;
    },

    stop() { teardown(); },
  };
}
