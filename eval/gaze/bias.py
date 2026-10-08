"""How much of the posture error is a constant shift (fixable by re-anchoring) vs. shape?"""
import numpy as np
exec(open("posture.py").read().split("POST = ")[0])
POST = ["lean_left", "lean_right", "far", "near"]
X = FEATS["iris+pose+lids"]; Xs = ema(X); Y = np.c_[tx, ty]
split = purs & (np.cumsum(purs) > purs.sum() / 2)
pred = PolyRidge(1).fit(X[cal0], Y[cal0], split[cal0]).predict(Xs)
rng = np.random.default_rng(0)
for p in POST + ["tiles", "fix2"]:
    sel = mask & (rec["phase"] == p)
    r = pred[sel] - Y[sel]
    b = r.mean(0)
    # realistic: estimate the shift from the FIRST 3 looks only (what 3 selections would teach it)
    segs = [s for s in dict.fromkeys(seg[sel])]
    first = sel & np.isin(seg, segs[:3]); rest = sel & np.isin(seg, segs[3:])
    b3 = (pred[first] - Y[first]).mean(0)
    print(f"{p:10s} raw err {np.abs(r).mean(0).round()}  shift {b.round()}  after removing shift {np.abs(r-b).mean(0).round()}  "
          f"| shift learned from 3 looks -> err on the rest {np.abs(pred[rest]-b3-Y[rest]).mean(0).round()}")
