// The eye -> screen model. Pure math, no DOM, so the browser and the offline checks
// (eval/gaze/parity.mjs) run exactly the same code.
//
// Chosen by measurement, not by guess (eval/gaze/, recording of 2026-10-07): on the 8 tile
// positions this gets 14/14 looks right, where the previous tracker got 6/14 on the same frames.
//
// Features per frame (14):
//   - each iris's position inside its eye, after putting the face into ONE head-fixed frame
//     (similarity Procrustes on bony landmarks) — 4 numbers
//   - the head pose that alignment needed: 3 rotation terms, log scale (distance), image x/y — 6
//   - eyelid shape, 4 principal components of the 32 lid-contour heights — looking down closes the
//     lid, and without this the model cannot tell a lowered eye from a lowered lid
// Model: plain ridge regression, one per axis. Linear beat every curved fit on held-out data.

export const RIGID = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 234, 454, 127, 356, 152, 33, 263, 133, 362];
export const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const N_LIDPC = 4;

/** Symmetric eigen-decomposition (cyclic Jacobi). Returns {values, vectors[col k]} sorted desc. */
export function eigSym(Ain) {
  const n = Ain.length;
  const A = Ain.map((r) => Float64Array.from(r));
  const V = Array.from({ length: n }, (_, i) => { const r = new Float64Array(n); r[i] = 1; return r; });
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => A[j][j] - A[i][i]);
  return { values: order.map((i) => A[i][i]), vectors: order.map((i) => V.map((r) => r[i])) };
}

/** Landmarks (normalised MediaPipe) -> pixel-scaled 3D points, so x and y share one unit. */
const toPx = (p, w, h) => [p.x * w, p.y * h, p.z * w];

/** Mean rigid shape over a set of frames, centred — the head-fixed frame everything maps into. */
export function makeReference(frames, w, h) {
  const ref = RIGID.map(() => [0, 0, 0]);
  for (const lm of frames) {
    const pts = RIGID.map((i) => toPx(lm[i], w, h));
    const c = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length);
    pts.forEach((p, j) => { for (let k = 0; k < 3; k++) ref[j][k] += (p[k] - c[k]) / frames.length; });
  }
  return ref;
}

/**
 * Head-fixed eye measurements for one frame: [iris L x,y, iris R x,y], pose[6], lid heights[32].
 * Horn's quaternion method finds the rotation Q with ref ≈ s·Q·(p − mu).
 */
export function rawFeatures(lm, w, h, ref) {
  const pts = RIGID.map((i) => toPx(lm[i], w, h));
  const mu = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length);
  const a = pts.map((p) => [p[0] - mu[0], p[1] - mu[1], p[2] - mu[2]]);
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let aa = 0;
  for (let j = 0; j < a.length; j++) {
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += a[j][r] * ref[j][c];
    aa += a[j][0] ** 2 + a[j][1] ** 2 + a[j][2] ** 2;
  }
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const N = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  const { vectors } = eigSym(N);
  const [q0, q1, q2, q3] = vectors[0];
  const Q = [
    [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
    [2 * (q2 * q1 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
    [2 * (q3 * q1 - q0 * q2), 2 * (q3 * q2 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
  ];
  let num = 0;
  for (let j = 0; j < a.length; j++) {
    for (let r = 0; r < 3; r++) num += ref[j][r] * (Q[r][0] * a[j][0] + Q[r][1] * a[j][1] + Q[r][2] * a[j][2]);
  }
  const s = num / aa;
  const align = (i) => {
    const p = toPx(lm[i], w, h);
    const d = [p[0] - mu[0], p[1] - mu[1], p[2] - mu[2]];
    return [0, 1, 2].map((r) => s * (Q[r][0] * d[0] + Q[r][1] * d[1] + Q[r][2] * d[2]));
  };
  const centroid = (idx) => {
    const P = idx.map(align);
    return [0, 1, 2].map((k) => P.reduce((t, p) => t + p[k], 0) / P.length);
  };
  const cl = centroid(LEFT_EYE), cr = centroid(RIGHT_EYE);
  const il = centroid([468, 469, 470, 471, 472]), ir = centroid([473, 474, 475, 476, 477]);
  const lids = [...LEFT_EYE.map((i) => align(i)[1] - cl[1]), ...RIGHT_EYE.map((i) => align(i)[1] - cr[1])];
  // Rotation terms: the row-vector R of the offline code is Qᵀ, so R[1][2] = Q[2][1], etc.
  const pose = [Q[2][1], Q[2][0], Q[1][0], Math.log(s), mu[0] / w - 0.5, mu[1] / h - 0.5];
  return { iris: [il[0] - cl[0], il[1] - cl[1], ir[0] - cr[0], ir[1] - cr[1]], pose, lids };
}

/** Lid PCA basis from calibration frames (centred on their mean). */
export function makeLidBasis(lidRows) {
  const d = lidRows[0].length, n = lidRows.length;
  const mean = new Array(d).fill(0);
  for (const r of lidRows) for (let k = 0; k < d; k++) mean[k] += r[k] / n;
  const C = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const r of lidRows) {
    for (let i = 0; i < d; i++) { const di = r[i] - mean[i]; for (let j = i; j < d; j++) C[i][j] += di * (r[j] - mean[j]); }
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) C[i][j] = C[j][i];
  const { vectors } = eigSym(C);
  return { mean, pcs: vectors.slice(0, N_LIDPC) };
}

/** The 14-number feature vector the model reads. */
export function featureVector(raw, lidBasis) {
  const lp = lidBasis.pcs.map((pc) => pc.reduce((s, v, k) => s + v * (raw.lids[k] - lidBasis.mean[k]), 0));
  return [...raw.iris, ...raw.pose, ...lp];
}

/** Ridge with an unpenalised intercept, on standardised features. */
function solve(A, b) {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-12) continue;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const w = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * w[j];
    w[i] = Math.abs(M[i][i]) < 1e-12 ? 0 : s / M[i][i];
  }
  return w;
}

export function fitRidge(X, Y, lambda) {
  const d = X[0].length;
  const mu = new Array(d).fill(0), sd = new Array(d).fill(0);
  for (const r of X) for (let k = 0; k < d; k++) mu[k] += r[k] / X.length;
  for (const r of X) for (let k = 0; k < d; k++) sd[k] += (r[k] - mu[k]) ** 2 / X.length;
  for (let k = 0; k < d; k++) sd[k] = Math.sqrt(sd[k]) || 1e-9;
  const Z = X.map((r) => [1, ...r.map((v, k) => (v - mu[k]) / sd[k])]);
  const A = Array.from({ length: d + 1 }, () => new Array(d + 1).fill(0));
  const bx = new Array(d + 1).fill(0), by = new Array(d + 1).fill(0);
  Z.forEach((z, i) => {
    for (let a = 0; a <= d; a++) {
      bx[a] += z[a] * Y[i][0]; by[a] += z[a] * Y[i][1];
      for (let b = a; b <= d; b++) A[a][b] += z[a] * z[b];
    }
  });
  for (let a = 0; a <= d; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
  for (let a = 1; a <= d; a++) A[a][a] += lambda;
  const wx = solve(A, bx), wy = solve(A, by);
  return {
    lambda,
    predict(x) {
      let px = wx[0], py = wy[0];
      for (let k = 0; k < d; k++) { const z = (x[k] - mu[k]) / sd[k]; px += wx[k + 1] * z; py += wy[k + 1] * z; }
      return [px, py];
    },
  };
}

/**
 * Fit from a calibration: still-dot samples (exact labels) + moving-dot samples (labels known
 * only up to the eye+camera delay). The delay and lambda are chosen by how well a model trained on
 * the MOVING dot predicts the STILL dots — those have no delay, so they are an honest referee.
 * Then the winner is refit on everything.
 *
 * still:  [{ f: number[14], x, y }]
 * moving: [{ f: number[14], t }]   with dotAt(t) -> [x, y] giving where the dot was at time t
 */
export function fitCalibration(still, moving, dotAt) {
  const err = (m, set) => set.reduce((s, p) => { const [x, y] = m.predict(p.f); return s + Math.hypot(x - p.x, y - p.y); }, 0) / set.length;
  let best = null;
  for (const lag of [100, 150, 200, 250, 300, 350, 400]) {
    const mv = moving.map((p) => { const [x, y] = dotAt(p.t - lag); return { f: p.f, x, y }; });
    for (const lambda of [10, 30, 100, 300, 1000]) {
      const m = fitRidge(mv.map((p) => p.f), mv.map((p) => [p.x, p.y]), lambda);
      const e = err(m, still);
      if (!best || e < best.e) best = { e, lag, lambda, mv };
    }
  }
  // Per-axis error on the still dots, for the app's "is this usable for these tiles" check.
  const mv0 = fitRidge(best.mv.map((p) => p.f), best.mv.map((p) => [p.x, p.y]), best.lambda);
  let ex = 0, ey = 0;
  for (const p of still) { const [x, y] = mv0.predict(p.f); ex += Math.abs(x - p.x); ey += Math.abs(y - p.y); }
  const all = [...still, ...best.mv];
  const model = fitRidge(all.map((p) => p.f), all.map((p) => [p.x, p.y]), best.lambda);
  return { model, samples: all, lag: best.lag, lambda: best.lambda, stillErrPx: Math.round(best.e),
    errX: Math.round(ex / still.length), errY: Math.round(ey / still.length) };
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
export function makeMedian(n = 7) {
  const bx = [], by = [];
  const mid = (a) => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)];
  const med = (x, y) => {
    bx.push(x); by.push(y);
    if (bx.length > n) { bx.shift(); by.shift(); }
    return [mid(bx), mid(by)];
  };
  // After a blink: fill the window with the point from before the lids moved, so the next frames
  // are judged against where he was looking, not against half-closed-lid frames.
  med.fill = (x, y) => { bx.length = 0; by.length = 0; for (let i = 0; i < n; i++) { bx.push(x); by.push(y); } };
  return med;
}

/**
 * Fixation detector. While the eye is moving, follow it fast. The moment it settles, LOCK —
 * and keep the dot dead still until it genuinely moves again.
 *
 * The lock is what makes the thing usable: a target that trembles under your gaze can never be
 * dwelled on, because every tremor resets the dwell.
 */
export function makeFixation({ moveThresh = 55, holdThresh = 32, settleMs = 120, breakMs = 0 } = {}) {
  let px = null, py = null;        // reported position
  let lx = 0, ly = 0;              // last raw
  let stillSince = 0, locked = false;
  let outSince = -1;               // when the point first left the lock radius (for breakMs)

  return (x, y, now) => {
    if (px === null) { px = x; py = y; lx = x; ly = y; stillSince = now; return [px, py, false]; }

    const step = Math.hypot(x - lx, y - ly);
    lx = x; ly = y;

    if (locked) {
      // Only break the lock on a real, SUSTAINED move: out of the radius for breakMs, not one noisy
      // frame. A saccade to another tile stays out; a noise spike comes straight back.
      if (Math.hypot(x - px, y - py) > moveThresh) {
        if (outSince < 0) outSince = now;
        if (now - outSince >= breakMs) { locked = false; stillSince = now; outSince = -1; }
        else return [px, py, true];
      } else { outSince = -1; return [px, py, true]; }
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
