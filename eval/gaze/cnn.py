"""Appearance model: a small CNN on the two 64x32 eye crops + head pose -> screen point.

Same training data and held-out sets as analyze.py, so the numbers are comparable.

    .venv/bin/python cnn.py [recording.jsonl]
"""
import sys

import numpy as np
import torch
import torch.nn as nn

from analyze import TEST_SETS, ema, lagged_targets, score
from load import aligned_eyes, fixation_mask, load

torch.manual_seed(0)


def prep(crops):
    """Per-crop contrast normalisation: the room light changes, the eye does not."""
    x = crops.astype(np.float32)
    mu = x.mean((1, 2), keepdims=True)
    sd = x.std((1, 2), keepdims=True) + 1e-3
    return (x - mu) / sd


class Net(nn.Module):
    def __init__(self, n_pose):
        super().__init__()
        def tower():
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 16x32
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 8x16
                nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 4x8
                nn.Flatten(), nn.Dropout(0.3), nn.Linear(48 * 4 * 8, 64), nn.ReLU())
        self.L, self.R = tower(), tower()
        self.head = nn.Sequential(nn.Linear(128 + n_pose, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, l, r, p):
        return self.head(torch.cat([self.L(l), self.R(r), p], 1))


def main():
    rec = load(sys.argv[1] if len(sys.argv) > 1 else None)
    n, W, H = rec["n"], rec["W"], rec["H"]
    mask, seg = fixation_mask(rec)
    blink = np.fmax(rec["bs"]["eyeBlinkLeft"], rec["bs"]["eyeBlinkRight"])
    purs = (rec["phase"] == "pursuit") & (blink < .6)
    fix1 = mask & (rec["phase"] == "fix1")
    lag = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    lx, ly = lagged_targets(rec, lag)
    cal = fix1 | purs
    Y = np.c_[np.where(fix1, rec["tx"], lx) / W, np.where(fix1, rec["ty"], ly) / H].astype(np.float32)

    _, _, pose = aligned_eyes(rec["lm"], rec["cam"]["w"], rec["cam"]["h"])
    pmu, psd = pose[cal].mean(0), pose[cal].std(0) + 1e-6
    P = ((pose - pmu) / psd).astype(np.float32)
    EL = torch.tensor(prep(rec["eL"]))[:, None]
    ER = torch.tensor(prep(rec["eR"]))[:, None]
    Pt, Yt = torch.tensor(P), torch.tensor(Y)

    idx = np.where(cal)[0]
    net = Net(P.shape[1])
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    for ep in range(60):
        net.train()
        perm = np.random.permutation(idx)
        for b in range(0, len(perm), 64):
            j = torch.tensor(perm[b:b + 64])
            l, r = EL[j], ER[j]
            # light augmentation: a pixel or two of crop jitter, brightness/contrast
            sh = np.random.randint(-2, 3, 2)
            l, r = torch.roll(l, tuple(sh), (2, 3)), torch.roll(r, tuple(sh), (2, 3))
            g = 1 + 0.1 * torch.randn(len(j), 1, 1, 1)
            loss = nn.functional.smooth_l1_loss(net(l * g, r * g, Pt[j]), Yt[j], beta=0.02)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        out = torch.cat([net(EL[i:i + 512], ER[i:i + 512], Pt[i:i + 512]) for i in range(0, n, 512)]).numpy()
    pred = ema(out * [W, H])
    score(f"5. CNN on eye images + head pose (labels shifted {lag} ms)", pred, rec, mask, seg)


if __name__ == "__main__":
    main()
