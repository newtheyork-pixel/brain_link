"""Geometric gaze model: a ray from the real head position, at head angle + eye angle, hits the
screen plane. Head translation and depth enter through physics, not through a learned fudge."""
import numpy as np
from scipy.optimize import least_squares
exec(open("posture.py").read().split("POST = ")[0])
POST = ["lean_left", "lean_right", "far", "near"]

R = rec["m"][:, :3, :3]; T = rec["m"][:, :3, 3]
# head forward direction in camera space (the face points along -z of the camera for a frontal face)
fwd = R[:, :, 2]                                # third column = head z axis
eye = np.c_[iris[:, [0, 1, 3, 4]], lids @ Vt.T]
mu, sd = eye[cal0].mean(0), eye[cal0].std(0) + 1e-9
E = (eye - mu) / sd
Es = ema(E); fwds = ema(fwd); Ts = ema(T)

def predict(p, E, fwd, T):
    k = E.shape[1]
    ax, ay = p[0] + E @ p[2:2 + k], p[1] + E @ p[2 + k:2 + 2 * k]     # eye-in-head angles (rad)
    g = p[2 + 2 * k]                                                   # head-rotation gain
    hx = np.arctan2(fwd[:, 0], fwd[:, 2]) * g
    hy = np.arctan2(fwd[:, 1], fwd[:, 2]) * g
    d = -T[:, 2]                                                        # distance to the screen plane
    xs = T[:, 0] + d * np.tan(hx + ax)
    ys = T[:, 1] + d * np.tan(hy + ay)
    kx, ky, cx, cy = p[3 + 2 * k:7 + 2 * k]
    return np.c_[cx + kx * xs, cy + ky * ys]

k = E.shape[1]
def fit(cal, lam=1.0):
    Y = np.c_[tx, ty][cal]
    p0 = np.r_[0, 0, np.zeros(2 * k), 1.0, -40, 40, W / 2, H / 2]
    def res(p):
        r = (predict(p, E[cal], fwd[cal], T[cal]) - Y).ravel() / 100
        return np.r_[r, lam * p[2:2 + 2 * k]]
    best = None
    for s in (1, -1):            # the camera mirror: try both signs of the screen x scale
        for s2 in (1, -1):
            q = p0.copy(); q[3 + 2 * k] *= s; q[4 + 2 * k] *= s2
            r = least_squares(res, q, loss="soft_l1", f_scale=1.0, max_nfev=4000)
            if best is None or r.cost < best.cost: best = r
    return best.x

def ev(pred, sel):
    hits = looks = 0
    for s in np.unique(seg[sel]):
        i = sel & (seg == s)
        hits += tile_of(np.array([np.median(pred[i, 0])]), np.array([np.median(pred[i, 1])]), W, H)[0] == tile_of(tx[i][:1], ty[i][:1], W, H)[0]
        looks += 1
    e = np.abs(pred[sel] - np.c_[tx, ty][sel]).mean(0)
    return f"err {e[0]:4.0f},{e[1]:4.0f}  {hits:2d}/{looks}"

for lam in ():
    p = fit(cal0, lam)
    pred = predict(p, Es, fwds, Ts)
    print(f"\nlam {lam}: normal calibration only   gain {p[2+2*k]:.2f} kx {p[3+2*k]:.0f} ky {p[4+2*k]:.0f}")
    for ph in ["fix2", "tiles"] + POST:
        print(f"   {ph:10s} {ev(pred, mask & (rec['phase'] == ph))}")

print("\n=== leave one posture out (calibration includes the other three postures)")
for held in POST:
    cal = cal0 | (mask & np.isin(rec["phase"], [q for q in POST if q != held]))
    p = fit(cal, 1)
    pred = predict(p, Es, fwds, Ts)
    print(f"  test {held:10s} {ev(pred, mask & (rec['phase'] == held))}   tiles {ev(pred, mask & (rec['phase'] == 'tiles'))}  gain {p[2+2*k]:.2f}")
