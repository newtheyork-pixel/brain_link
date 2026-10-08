"""Score gaze models against one recording, on the same held-out data every time.

    .venv/bin/python analyze.py [recording.jsonl]

Training data = what a calibration could collect: the first dot grid (fix1) and/or the moving
dot (pursuit). Everything else is held out:
  fix2         the same 20 dots, later, same posture       -> pure accuracy
  lean/far/near  the same dots after he moved               -> does it survive posture
  tiles        the 8 tile centres of the real app layout    -> the number that matters
"""
import sys

import numpy as np

from load import load, fixation_mask, js_features, aligned_eyes

TEST_SETS = {"fix2": ["fix2"], "posture": ["lean_left", "lean_right", "far", "near"], "tiles": ["tiles"]}


def ema(X, a=0.30):
    """The app's causal feature smoothing (gaze.js smoothFeatures)."""
    Y = X.copy()
    for i in range(1, len(X)):
        Y[i] = Y[i - 1] + a * (X[i] - Y[i - 1])
    return Y


def lagged_targets(rec, lag_ms):
    """Where the dot was `lag_ms` before each frame: the eye (and the camera) trail the dot."""
    t = rec["t"]
    ok = ~np.isnan(rec["tx"])
    return (np.interp(t - lag_ms, t[ok], rec["tx"][ok]), np.interp(t - lag_ms, t[ok], rec["ty"][ok]))


def tile_of(x, y, W, H):
    return (np.clip((x / (W / 4)).astype(int), 0, 3) + 4 * np.clip((y / (H / 2)).astype(int), 0, 1))


# ------------------------------------------------------------------ models


def ridge(X, y, lam):
    Xb = np.c_[np.ones(len(X)), X]
    A = Xb.T @ Xb + lam * np.diag([0] + [1] * X.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


class JSModel:
    """A port of gaze.js buildModel: 5x5 RBF on the first two features + linear terms, ridge,
    feature-variant and lambda chosen on a held-out pass."""
    X_VAR = {"iris": ["ix", "iy"], "iris+head": ["ix", "iy", "yaw", "pitch"], "iris+head+pos": ["ix", "iy", "yaw", "nx", "sc"]}
    Y_VAR = {"iris": ["iy", "ix"], "iris+lid": ["iy", "ix", "ap"], "iris+lid+head": ["iy", "ix", "ap", "pitch"],
             "iris+lid+head+pos": ["iy", "ix", "ap", "pitch", "ny", "sc"]}

    @staticmethod
    def basis(F):
        mu, sg = F.mean(0), F.std(0) + 1e-4
        Z = (F - mu) / sg
        lo0, hi0 = np.quantile(Z[:, 0], [.05, .95])
        lo1, hi1 = np.quantile(Z[:, 1], [.05, .95])
        s0, s1 = max(.5, hi0 - lo0), max(.5, hi1 - lo1)
        C = np.array([[lo0 + s0 * i / 4, lo1 + s1 * j / 4] for i in range(5) for j in range(5)])
        g = 1 / (2 * (max(s0, s1) / 4) ** 2)

        def feat(G):
            Zg = (G - mu) / sg
            d2 = ((Zg[:, None, :2] - C[None]) ** 2).sum(-1)
            return np.c_[Zg, np.exp(-g * d2)]
        return feat

    def fit(self, feats, tx, ty, split):
        self.ax = {}
        for axis, target, variants in (("x", tx, self.X_VAR), ("y", ty, self.Y_VAR)):
            best = None
            tr, te = ~split, split
            for name, cols in variants.items():
                F = np.c_[[feats[c] for c in cols]].T
                b = self.basis(F[tr])
                for lam in [.3, 1, 3, 10, 30, 100]:
                    w = ridge(b(F[tr]), target[tr], lam)
                    err = np.abs(np.c_[np.ones(te.sum()), b(F[te])] @ w - target[te]).mean()
                    if best is None or err < best[0]:
                        best = (err, cols, lam)
            _, cols, lam = best
            F = np.c_[[feats[c] for c in cols]].T
            b = self.basis(F)
            self.ax[axis] = (cols, b, ridge(b(F), target, lam))
        return self

    def predict(self, feats, idx):
        out = []
        for axis in "xy":
            cols, b, w = self.ax[axis]
            F = np.c_[[feats[c][idx] for c in cols]].T
            out.append(np.c_[np.ones(len(F)), b(F)] @ w)
        return np.array(out).T


class PolyRidge:
    """Ridge on a quadratic expansion of a feature matrix, lambda by held-out error."""
    def __init__(self, deg=2):
        self.deg = deg

    def expand(self, X):
        Z = (X - self.mu) / self.sg
        cols = [Z]
        if self.deg >= 2:
            iu = np.triu_indices(Z.shape[1])
            cols.append((Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]])
        return np.concatenate(cols, 1)

    def fit(self, X, Y, split):
        self.mu, self.sg = X.mean(0), X.std(0) + 1e-9
        E = self.expand(X)
        best = None
        for lam in [1e-2, 1e-1, 1, 3, 10, 30, 100, 300, 1000]:
            W = ridge(E[~split], Y[~split], lam)
            err = np.abs(np.c_[np.ones(split.sum()), E[split]] @ W - Y[split]).mean()
            if best is None or err < best[0]:
                best = (err, lam)
        self.lam = best[1]
        self.W = ridge(E, Y, self.lam)
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), self.expand(X)] @ self.W


# ------------------------------------------------------------------ scoring


def score(name, pred, rec, mask, seg):
    """pred: (n,2) predicted screen px for every frame (NaN where not predicted)."""
    W, H = rec["W"], rec["H"]
    rows = []
    for set_name, phases in TEST_SETS.items():
        sel = mask & np.isin(rec["phase"], phases) & ~np.isnan(pred[:, 0])
        if not sel.any():
            continue
        ex = np.abs(pred[sel, 0] - rec["tx"][sel]).mean()
        ey = np.abs(pred[sel, 1] - rec["ty"][sel]).mean()
        # One decision per dot: the median over the fixation, as dwell would see it.
        hits, n = 0, 0
        for s in np.unique(seg[sel]):
            i = sel & (seg == s)
            px, py = np.median(pred[i, 0]), np.median(pred[i, 1])
            hits += tile_of(np.array([px]), np.array([py]), W, H)[0] == tile_of(rec["tx"][i][:1], rec["ty"][i][:1], W, H)[0]
            n += 1
        # Per frame: how often the gaze point is on the right tile at any instant.
        fr = (tile_of(pred[sel, 0], pred[sel, 1], W, H) == tile_of(rec["tx"][sel], rec["ty"][sel], W, H)).mean()
        rows.append(f"{set_name}: err x {ex:4.0f} y {ey:4.0f}px  tile {hits}/{n} per-look, {fr:4.0%} per-frame")
    print(f"\n{name}\n   " + "\n   ".join(rows))


def main():
    rec = load(sys.argv[1] if len(sys.argv) > 1 else None)
    meta = rec["meta"]
    print(f"{rec['path'].name}: {rec['n']} frames, window {rec['W']}x{rec['H']}, camera {rec['cam']}")
    for p in dict.fromkeys(rec["phase"]):
        print(f"   {p:11s} {np.sum(rec['phase'] == p):5d} frames")
    dt = np.diff(rec["t"])
    print(f"   frame interval median {np.median(dt):.0f} ms")

    mask, seg = fixation_mask(rec)
    jf = js_features(rec["lm"], rec["m"])
    jf_s = {k: ema(v) for k, v in jf.items()}           # as the app sees them at runtime
    purs = rec["phase"] == "pursuit"
    fix1 = mask & (rec["phase"] == "fix1")
    n = rec["n"]
    half = np.zeros(n, bool)
    half[np.where(purs)[0][len(np.where(purs)[0]) // 2:]] = True

    # 1. The app today: calibrate on the moving dot, labels with no delay correction.
    tr = purs & (np.fmax(rec["bs"]["eyeBlinkLeft"], rec["bs"]["eyeBlinkRight"]) < .6) & ~np.isnan(rec["tx"])
    feats_tr = {k: v[tr] for k, v in jf.items()}
    m = JSModel().fit(feats_tr, rec["tx"][tr], rec["ty"][tr], half[tr])
    pred = np.full((n, 2), np.nan)
    pred[:] = m.predict(jf_s, np.arange(n))
    score("1. TODAY'S TRACKER (moving-dot calibration, no delay fix)", pred, rec, mask, seg)

    # 2. How late is the eye+camera behind the dot? Pick the lag that best fits the pursuit.
    best = None
    for lag in range(0, 401, 25):
        lx, ly = lagged_targets(rec, lag)
        mm = JSModel().fit(feats_tr, lx[tr], ly[tr], half[tr])
        p = mm.predict({k: v[tr] for k, v in jf.items()}, np.arange(tr.sum()))
        e = np.abs(p[:, 0] - lx[tr]).mean() + np.abs(p[:, 1] - ly[tr]).mean()
        if best is None or e < best[0]:
            best = (e, lag)
    lag = best[1]
    print(f"\n   eye+camera delay behind the moving dot: ~{lag} ms")
    lx, ly = lagged_targets(rec, lag)
    m = JSModel().fit(feats_tr, lx[tr], ly[tr], half[tr])
    pred[:] = m.predict(jf_s, np.arange(n))
    score(f"2. same, labels shifted by {lag} ms", pred, rec, mask, seg)

    # 3. Calibrate on still dots instead (no delay problem at all), + delay-fixed pursuit.
    cal = fix1 | tr
    cx, cy = np.where(fix1, rec["tx"], lx), np.where(fix1, rec["ty"], ly)
    split = purs & half                     # lambda/variant chosen on the late half of the pursuit
    m = JSModel().fit({k: v[cal] for k, v in jf.items()}, cx[cal], cy[cal], split[cal])
    pred[:] = m.predict(jf_s, np.arange(n))
    score("3. today's model, still dots + delay-fixed moving dot", pred, rec, mask, seg)

    # 4. Better features: head-fixed iris position for EACH eye, lid shape, full head pose.
    iris, contour, pose = aligned_eyes(rec["lm"], rec["cam"]["w"], rec["cam"]["h"])
    lids = contour[:, 1::2]                                  # y of every contour point = lid shape
    sets = {
        "iris (both eyes, head-fixed)": iris[:, [0, 1, 3, 4]],
        "+ head pose": np.c_[iris[:, [0, 1, 3, 4]], pose],
        "+ head pose + lids": np.c_[iris[:, [0, 1, 3, 4]], pose, lids @ np.linalg.svd(lids - lids.mean(0), full_matrices=False)[2][:4].T],
    }
    for name, X in sets.items():
        Xs = ema(X)
        Y = np.c_[cx, cy]
        for deg in (1, 2):
            pr = PolyRidge(deg).fit(X[cal], Y[cal], split[cal])
            pred[:] = pr.predict(Xs)
            score(f"4. {name}, degree {deg}, lambda {pr.lam}", pred, rec, mask, seg)


if __name__ == "__main__":
    main()
