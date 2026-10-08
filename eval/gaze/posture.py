"""Does calibrating with head movement fix the posture failure? Leave one posture out."""
import numpy as np
from load import load, fixation_mask, aligned_eyes, js_features
from analyze import ema, lagged_targets, tile_of, PolyRidge

rec = load(); n, W, H = rec["n"], rec["W"], rec["H"]
mask, seg = fixation_mask(rec)
blink = np.fmax(rec["bs"]["eyeBlinkLeft"], rec["bs"]["eyeBlinkRight"])
purs = (rec["phase"] == "pursuit") & (blink < .6) & ~np.isnan(rec["tx"])
lx, ly = lagged_targets(rec, 175)
tx = np.where(purs, lx, rec["tx"]); ty = np.where(purs, ly, rec["ty"])
iris, contour, pose = aligned_eyes(rec["lm"], rec["cam"]["w"], rec["cam"]["h"])
lids = contour[:, 1::2]
cal0 = (mask & (rec["phase"] == "fix1")) | purs
Vt = np.linalg.svd(lids[cal0] - lids[cal0].mean(0), full_matrices=False)[2][:4]
jf = js_features(rec["lm"], rec["m"])
FEATS = {
  "iris+pose+lids": np.c_[iris[:, [0, 1, 3, 4]], pose, lids @ Vt.T],
  "iris(3d)+pose+lids": np.c_[iris, pose, lids @ Vt.T],
  "iris+pose+lids+head-matrix": np.c_[iris[:, [0, 1, 3, 4]], pose, lids @ Vt.T, rec["m"][:, :3, :].reshape(n, -1)],
}
POST = ["lean_left", "lean_right", "far", "near"]

def evaluate(pred, sel):
    hits = looks = 0
    for s in np.unique(seg[sel]):
        i = sel & (seg == s)
        hits += tile_of(np.array([np.median(pred[i, 0])]), np.array([np.median(pred[i, 1])]), W, H)[0] == tile_of(tx[i][:1], ty[i][:1], W, H)[0]
        looks += 1
    e = np.abs(pred[sel] - np.c_[tx, ty][sel]).mean(0)
    return f"err {e[0]:4.0f},{e[1]:4.0f}  {hits:2d}/{looks}"

for name, X in FEATS.items():
    Xs = ema(X); Y = np.c_[tx, ty]
    print(f"\n== {name}")
    for held in POST:
        for label, extra in (("normal calib only     ", []), ("calib + other postures", [p for p in POST if p != held])):
            cal = cal0 | (mask & np.isin(rec["phase"], extra))
            split = purs & (np.cumsum(purs) > purs.sum() / 2)
            pr = PolyRidge(1).fit(X[cal], Y[cal], split[cal])
            pred = pr.predict(Xs)
            sel = mask & (rec["phase"] == held)
            tiles = mask & (rec["phase"] == "tiles")
            print(f"  test {held:10s} {label}: {evaluate(pred, sel)}   tiles {evaluate(pred, tiles)}")
