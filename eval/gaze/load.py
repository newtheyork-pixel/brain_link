"""Load a /record.html recording and turn each frame into gaze features.

A recording is data/recordings/<id>.jsonl: one `meta` line, then one `f` line per camera frame
(478 landmarks, eye blendshapes, head matrix, 64x32 grayscale crop of each eye, the dot position).
"""
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REC_DIR = ROOT / "data" / "recordings"

# Same indices as public/gaze.js.
L_OUT, L_IN, L_TOP, L_BOT = 33, 133, 159, 145
R_IN, R_OUT, R_TOP, R_BOT = 362, 263, 386, 374
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Bony points that do not move with the eyes, lids or mouth: forehead, nose bridge, nose tip,
# cheekbones, temples, chin. Used to put every frame into one head-fixed frame.
RIGID = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 234, 454, 127, 356, 152, 33, 263, 133, 362]


def latest():
    files = sorted(REC_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("no recordings in data/recordings/ yet")
    return files[-1]


def load(path=None):
    path = Path(path) if path else latest()
    meta, rows = None, []
    for line in path.open():
        d = json.loads(line)
        if d["k"] == "meta":
            meta = d
        elif d["k"] == "f":
            rows.append(d)
    if meta is None:
        raise SystemExit(f"{path.name}: no meta line")
    n = len(rows)
    lm = np.array([r["lm"] for r in rows], dtype=np.float64).reshape(n, 478, 3) / 1e5
    W, H = meta["win"]["w"], meta["win"]["h"]
    tx = np.array([np.nan if r["tx"] is None else r["tx"] * W for r in rows])
    ty = np.array([np.nan if r["ty"] is None else r["ty"] * H for r in rows])
    m = np.array([r["m"] if r["m"] else [np.nan] * 16 for r in rows]).reshape(n, 4, 4).transpose(0, 2, 1)  # column-major
    bs_keys = sorted({k for r in rows for k in r["bs"]})
    bs = {k: np.array([r["bs"].get(k, np.nan) for r in rows]) for k in bs_keys}

    def crops(key):
        cw, ch = meta["crop"]
        return np.stack([np.frombuffer(base64.b64decode(r[key]), dtype=np.uint8).reshape(ch, cw) for r in rows])

    return {
        "path": path, "meta": meta, "W": W, "H": H, "n": n,
        "t": np.array([r["t"] for r in rows]), "phase": np.array([r["ph"] for r in rows]),
        "tx": tx, "ty": ty, "lm": lm, "m": m, "bs": bs,
        "eL": crops("eL"), "eR": crops("eR"),
        "cam": meta["cam"],
    }


def segments(rec):
    """Split into presentations: runs of frames with the same phase and the same dot position."""
    seg = np.zeros(rec["n"], dtype=int)
    k = 0
    for i in range(1, rec["n"]):
        same = (rec["phase"][i] == rec["phase"][i - 1]
                and np.isclose(rec["tx"][i], rec["tx"][i - 1], equal_nan=True)
                and np.isclose(rec["ty"][i], rec["ty"][i - 1], equal_nan=True))
        if not same or rec["phase"][i] == "pursuit":
            k += 1
        seg[i] = k
    return seg


def fixation_mask(rec, settle_ms=500):
    """Frames where he was holding still on a dot: drop the first `settle_ms` after it appeared
    (saccade + latency), blinks, and the message screens."""
    seg = segments(rec)
    t = rec["t"]
    keep = np.zeros(rec["n"], dtype=bool)
    blink = np.fmax(rec["bs"].get("eyeBlinkLeft", 0), rec["bs"].get("eyeBlinkRight", 0))
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        if rec["phase"][idx[0]] == "pursuit" or np.isnan(rec["tx"][idx[0]]):
            continue
        keep[idx[t[idx] - t[idx[0]] >= settle_ms]] = True
    keep &= blink < 0.5
    return keep, seg


# ---------------------------------------------------------------- features


def js_features(lm, m):
    """Exactly what public/gaze.js computes today: [ix, iy, yaw, pitch, ap, nx, ny, sc]."""
    def sub(a, b): return a - b
    def norm(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    P = lambda i: lm[:, i, :]
    face_down = norm(P(152) - P(10))

    def eye(ia, iz, c1, c2):
        a, c = P(c1), P(c2)
        ex = norm(c - a)
        ey = norm(face_down - ex * np.sum(face_down * ex, -1, keepdims=True))
        w = np.linalg.norm(c - a, axis=-1) + 1e-6
        mid = (a + c) / 2
        d = lm[:, ia:iz + 1, :].mean(1) - mid
        return np.sum(d * ex, -1) / w, np.sum(d * ey, -1) / w

    Lx, Ly = eye(468, 472, L_OUT, L_IN)
    Rx, Ry = eye(473, 477, R_IN, R_OUT)
    ap_of = lambda t, b, c1, c2: np.linalg.norm(P(b) - P(t), axis=-1) / (np.linalg.norm(P(c2) - P(c1), axis=-1) + 1e-6)
    ap = (ap_of(L_TOP, L_BOT, L_OUT, L_IN) + ap_of(R_TOP, R_BOT, R_IN, R_OUT)) / 2
    yaw = np.arctan2(-m[:, 2, 0], np.hypot(m[:, 2, 1], m[:, 2, 2]))   # m[8], m[9], m[10] column-major
    pitch = np.arctan2(m[:, 2, 1], m[:, 2, 2])
    nx, ny = P(1)[:, 0] - 0.5, P(1)[:, 1] - 0.5
    sc = np.log(np.linalg.norm(P(L_OUT) - P(R_OUT), axis=-1) + 1e-6)
    return {"ix": (Lx + Rx) / 2, "iy": (Ly + Ry) / 2, "yaw": yaw, "pitch": pitch, "ap": ap,
            "nx": nx, "ny": ny, "sc": sc, "Lx": Lx, "Ly": Ly, "Rx": Rx, "Ry": Ry}


def aligned_eyes(lm, cam_w, cam_h):
    """Put every frame into one head-fixed frame (similarity Procrustes on the rigid points, in
    pixel units so x and y are on the same scale), then read the iris against the eye contour.

    Returns per frame: each iris centre relative to its eye-contour centroid, the eye contours
    themselves (lid shape), and the alignment's rotation/scale/translation (head pose + position).
    """
    px = lm.copy()
    px[..., 0] *= cam_w
    px[..., 1] *= cam_h
    px[..., 2] *= cam_w       # MediaPipe z is in the same scale as x
    ref = px[:, RIGID, :].mean(0)
    ref = ref - ref.mean(0)
    n = len(px)
    out_iris, out_contour, pose = [], [], []
    for i in range(n):
        A = px[i, RIGID, :]
        mu = A.mean(0)
        A0 = A - mu
        U, S, Vt = np.linalg.svd(A0.T @ ref)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ D @ Vt                      # rotation taking this frame onto the reference
        s = (S * np.diag(D)).sum() / (A0 ** 2).sum()
        al = s * (px[i] - mu) @ R          # every landmark, head-fixed
        cl, cr = al[LEFT_EYE].mean(0), al[RIGHT_EYE].mean(0)
        il, ir = al[468:473].mean(0), al[473:478].mean(0)
        out_iris.append(np.r_[il - cl, ir - cr])
        out_contour.append(np.r_[(al[LEFT_EYE] - cl)[:, :2].ravel(), (al[RIGHT_EYE] - cr)[:, :2].ravel()])
        # pose: rotation as 3 numbers (small-angle), log scale, head position in the image
        pose.append([R[1, 2], R[0, 2], R[0, 1], np.log(s), mu[0] / cam_w - 0.5, mu[1] / cam_h - 0.5])
    return np.array(out_iris), np.array(out_contour), np.array(pose)
