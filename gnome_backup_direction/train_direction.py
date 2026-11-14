"""Direction task: signals -> adapted EQNet -> probability map of the wall direction (256x256, 1 channel).
Target = Gaussian heat map on the sphere (sigma 2 deg) from meta.csv, zeros for noise. Repo root:
    python -m gnome.train_direction [--epochs 20]
Metric: angle between the argmax pixel and the true direction after folding hemispheres (as datasets/code/evaluate.py);
hit = within 15 deg. Compared with the classical baseline columns of meta.csv and a random pixel."""
import argparse
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gnome.geometry import sensor_features
from gnome.loader import DEFAULT_CSV, GnomeSignals, valid_mask
from gnome.model import GnomeEQNet

sys.path.insert(0, str(DEFAULT_CSV.parent / "code"))
import dh_core as dh  # noqa: E402

KINDS = ["wall", "ladder", "noise"]  # FALSE left out (README)


def load(ds, ids):
    items = [ds.get(i) for i in ids]
    return {
        "data": torch.stack([i["data"] for i in items]),
        "heatmap": torch.stack([i["heatmap"] for i in items]),
        "dont_care": torch.tensor([i["dont_care"] for i in items]),
        "ids": ids,
    }


def weighted_bce(logits, hm, valid, keep, alpha=50.0):
    """BCE over the disk; peak pixels weighted (1 + alpha*heat); don't-care samples ignored."""
    w = (1.0 + alpha * hm) * valid * keep.float()[:, None, None, None]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, hm, weight=w, reduction="sum")
    return loss / (valid.sum().item() * max(keep.sum().item(), 1))


def softmax_ce(logits, hm, valid, keep):
    """Cross-entropy of the softmax over disk pixels against the normalized heat map.
    A distribution over pixels cannot collapse to 'all zeros'. Only wall samples (hm.sum() > 0) are used."""
    lg = logits[:, 0].masked_fill(valid[0, 0] == 0, -1e9).flatten(1)
    t = hm[:, 0].flatten(1)
    use = keep & (t.sum(1) > 0)
    if use.sum() == 0:
        return logits.sum() * 0.0
    t = t[use] / t[use].sum(1, keepdim=True)
    return -(t * torch.log_softmax(lg[use], 1)).sum(1).mean()


def collapse_check(logits, d, valid):
    """Is the net just emitting empty images? Compare with the best constant predictor and look at the peaks."""
    p = torch.sigmoid(logits) * valid
    hm = d["heatmap"]
    wall = (hm.flatten(1).amax(1) > 0.5) & ~d["dont_care"]
    pk, bg = (hm > 0.5) & wall[:, None, None, None], (hm < 0.01) & wall[:, None, None, None] & (valid > 0)
    p_peak, p_bg = p[pk].mean().item(), p[bg].mean().item()
    mean_hm = hm[wall].mean(0, keepdim=True).expand_as(hm)  # best constant image
    ok = (valid > 0).expand_as(hm)
    bce = lambda q: torch.nn.functional.binary_cross_entropy(q.clamp(1e-6, 1 - 1e-6)[ok & wall[:, None, None, None]],
                                                             hm[ok & wall[:, None, None, None]]).item()
    pos = torch.stack([(p[i, 0].flatten().argmax()) for i in torch.where(wall)[0]])
    print(f"  collapse check (val walls): p(true peak) {p_peak:.4f} vs p(background) {p_bg:.4f} "
          f"(x{p_peak / max(p_bg, 1e-9):.1f}) | max p per image: mean {p[wall].flatten(1).amax(1).mean():.3f}")
    print(f"  distinct argmax positions {len(pos.unique())}/{len(pos)} (1 = same pixel for every sample) | "
          f"BCE model {bce(p):.4f} vs constant mean-heatmap {bce(mean_hm * valid):.4f}")


def predict(model, d, bs=16):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(d["ids"]), bs):
            out.append(model({"data": d["data"][i : i + bs]})["hough"])
    return torch.cat(out)  # logits [N, 1, 256, 256]


def angles(logits, ids, meta, valid):
    """argmax pixel -> direction -> folded angle to the truth [deg]; max probability as score."""
    lg = logits[:, 0].masked_fill(valid[0, 0] == 0, -1e9)
    flat = lg.flatten(1)
    idx = flat.argmax(1)
    score = torch.sigmoid(flat.max(1).values).numpy()
    y, x = (idx // 256).numpy(), (idx % 256).numpy()
    pred = dh.pixel_to_xyz(x, y)
    truth = meta.loc[ids, ["dir_x", "dir_y", "dir_z"]].values
    return dh.folded_angle_deg(pred, truth), score


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1.0
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def report(name, logits, d, meta, valid):
    ids = d["ids"]
    ang, score = angles(logits, ids, meta, valid)
    tgt = np.array([meta.kind[i] != "noise" and not dc for i, dc in zip(ids, d["dont_care"].tolist())])
    neg = np.array([meta.kind[i] == "noise" for i in ids])
    a = ang[tgt]
    print(f"[{name}] targets {tgt.sum()}  recall@15deg {np.mean(a <= 15):.2f}  median {np.median(a):.1f} deg  "
          f"p90 {np.percentile(a, 90):.1f} deg  | AUC wall vs noise {auc(score[tgt], score[neg]):.3f}")
    return ang, tgt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="gnome/train_direction.png")
    p.add_argument("--loss", choices=["bce", "softmax"], default="bce")
    p.add_argument("--head", choices=["dense", "geo"], default="geo")
    p.add_argument("--alpha", type=float, default=50.0, help="peak weight in the BCE")
    a = p.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    ds = GnomeSignals(with_stacks=False)
    meta = ds.meta
    valid = valid_mask()[None, None]
    tr, va = load(ds, ds.ids("train", KINDS)), load(ds, ds.ids("val", KINDS))
    print(f"train {len(tr['ids'])}  val {len(va['ids'])}  (don't care: {int(tr['dont_care'].sum())}/{int(va['dont_care'].sum())})")

    if a.head == "geo":
        model = GnomeEQNet(head="geo", n_out=1, geometry=sensor_features(ds.sensors))
    else:
        model = GnomeEQNet(head="dense", n_out=1)
    print(f"head {a.head}: new-head params {sum(p.numel() for p in model.hough_head.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    n = len(tr["ids"])
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        run = 0.0
        for i in range(0, n, a.bs):
            j = perm[i : i + a.bs]
            logits = model({"data": tr["data"][j]})["hough"]
            if a.loss == "bce":
                loss = weighted_bce(logits, tr["heatmap"][j], valid, ~tr["dont_care"][j], a.alpha)
            else:
                loss = softmax_ce(logits, tr["heatmap"][j], valid, ~tr["dont_care"][j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(j)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep + 1:2d} train loss {run / n:.5f} ({time.time() - t0:.0f}s)")

    lv = predict(model, va)
    ang, tgt = report("val model   ", lv, va, meta, valid)
    collapse_check(lv, va, valid)

    # baselines on the same val targets
    vid = np.array(va["ids"])[tgt]
    b = meta.loc[vid]
    base_ok = (b.base_detected == 1) & (b.base_angle_err_deg <= 15)
    print(f"[val classical] recall@15deg {base_ok.mean():.2f}  median {b.base_angle_err_deg.median():.1f} deg "
          f"(meta.csv base_*; detections on the strongest channel)")
    rnd = dh.folded_angle_deg(dh.pixel_to_xyz(*np.random.default_rng(0).integers(40, 216, (2, len(vid))).tolist()),
                              b[["dir_x", "dir_y", "dir_z"]].values)
    print(f"[val random px] recall@15deg {np.mean(rnd <= 15):.2f}")

    k = [i for i, m in enumerate(va["ids"]) if tgt[i]][:3]
    fig, ax = plt.subplots(2, len(k), figsize=(3.4 * len(k), 6.6), squeeze=False)
    for c, i in enumerate(k):
        ax[0, c].imshow(va["heatmap"][i, 0], cmap="viridis")
        ax[0, c].set_title(f"#{va['ids'][i]} target", fontsize=9)
        ax[1, c].imshow(torch.sigmoid(lv[i, 0]) * valid[0, 0], cmap="viridis")
        ax[1, c].set_title(f"prediction, error {ang[i]:.0f} deg", fontsize=9)
    fig.suptitle("direction map: short training on the pilot subset", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
