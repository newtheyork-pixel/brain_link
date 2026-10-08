import numpy as np
exec(open("posture.py").read().split("POST = ")[0])
X = FEATS["iris+pose+lids"]; Xs = ema(X); Y = np.c_[tx, ty]
fix1 = mask & (rec["phase"] == "fix1")
# 9-dot subset of fix1: the grid points nearest a 3x3 layout
g = np.c_[rec["tx"], rec["ty"]] / [W, H]
nine = fix1 & np.isin(np.round(g[:, 0], 2), [0.06, 0.5, 0.94]) & np.isin(np.round(g[:, 1], 2), [0.08, 0.36, 0.92])
def ev(pred, ph):
    sel = mask & (rec["phase"] == ph); hits = looks = 0
    for s in np.unique(seg[sel]):
        i = sel & (seg == s)
        hits += tile_of(np.array([np.median(pred[i, 0])]), np.array([np.median(pred[i, 1])]), W, H)[0] == tile_of(tx[i][:1], ty[i][:1], W, H)[0]; looks += 1
    e = np.abs(pred[sel] - Y[sel]).mean(0)
    return f"{e[0]:4.0f},{e[1]:4.0f} {hits:2d}/{looks}"
half = purs & (np.cumsum(purs) > purs.sum() / 2)
for name, cal, split in [("moving dot only (delay-fixed)", purs, half), ("20 still dots only", fix1, fix1 & (np.cumsum(fix1) > fix1.sum()*.7)),
                         ("9 still dots only", nine, nine & (np.cumsum(nine) > nine.sum()*.7)),
                         ("20 dots + moving dot", fix1 | purs, half), ("9 dots + moving dot", nine | purs, half)]:
    pr = PolyRidge(1).fit(X[cal], Y[cal], split[cal]); pred = pr.predict(Xs)
    print(f"{name:32s} fix2 {ev(pred,'fix2')}   tiles {ev(pred,'tiles')}   lam {pr.lam}")
# lag sensitivity
for lag in (0, 100, 175, 250):
    lx2, ly2 = lagged_targets(rec, lag); Y2 = Y.copy(); Y2[purs] = np.c_[lx2, ly2][purs]
    pr = PolyRidge(1).fit(X[purs], Y2[purs], half[purs]); pred = pr.predict(Xs)
    print(f"moving dot only, delay {lag:3d} ms    fix2 {ev(pred,'fix2')}   tiles {ev(pred,'tiles')}")
