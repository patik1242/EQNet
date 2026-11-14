"""Short smoke test: can the adapted EQNet *learn* signals -> Hough accumulator?  Repo root:
    python -m gnome.train_smoke [--n_train 96] [--epochs 10]
Not a real training: a few minutes on CPU, a small train subset, compared with the trivial baseline "predict 0"
and "predict the mean train accumulator" on the val split."""
import argparse
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gnome.loader import GnomeSignals, valid_mask
from gnome.model import GnomeEQNet

KINDS = ["wall", "ladder", "noise"]  # README: leave FALSE out at first


def masked_mse(pred, target, mask):
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() * pred.shape[0] * pred.shape[1])


def evaluate(model, ds, ids, mask, bs, mean_acc):
    model.eval()
    tot = {"model": 0.0, "zero": 0.0, "mean": 0.0}
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            b = ds.batch(ids[i : i + bs])
            y = b["target"]
            n = len(y)
            tot["model"] += masked_mse(model(b)["hough"], y, mask).item() * n
            tot["zero"] += masked_mse(torch.zeros_like(y), y, mask).item() * n
            tot["mean"] += masked_mse(mean_acc.expand_as(y), y, mask).item() * n
    return {k: v / len(ids) for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=96)
    p.add_argument("--n_val", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="gnome/train_smoke.png")
    a = p.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    ds = GnomeSignals()
    assert ds.meta is not None and ds._stacks_path is not None, "needs datasets/meta.csv and datasets/stacks.h5"
    train = random.sample(ds.ids("train", KINDS), a.n_train)
    val = random.sample(ds.ids("val", KINDS), a.n_val)
    mask = valid_mask()[None, None]
    mean_acc = torch.stack([ds.target(i) for i in train]).mean(0, keepdim=True)

    model = GnomeEQNet(head="dense")
    n_new = sum(p.numel() for p in model.hough_head.parameters())
    n_all = sum(p.numel() for p in model.backbone.parameters())
    print(f"train {len(train)}  val {len(val)}  | backbone params {n_all:,}  new head params {n_new:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        random.shuffle(train)
        run = 0.0
        for i in range(0, len(train), a.bs):
            b = ds.batch(train[i : i + a.bs])
            loss = masked_mse(model(b)["hough"], b["target"], mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(b["target"])
        print(f"epoch {ep + 1:2d}  train mse {run / len(train):.4f}  ({time.time() - t0:.0f}s)")

    r = evaluate(model, ds, val, mask, a.bs, mean_acc)
    print("val mse  model {model:.4f} | predict-0 {zero:.4f} | predict-mean-train {mean:.4f}".format(**r))

    # image: strongest-target channel of 3 val walls: target vs prediction
    wall = [i for i in val if ds.meta.kind[i] == "wall"][:3] or val[:3]
    b = ds.batch(wall)
    model.eval()
    with torch.no_grad():
        pred = model(b)["hough"]
    fig, ax = plt.subplots(2, len(wall), figsize=(3.4 * len(wall), 6.6), squeeze=False)
    for j, sid in enumerate(wall):
        k = int(b["target"][j].amax((1, 2)).argmax())
        ax[0, j].imshow(b["target"][j, k], cmap="viridis")
        ax[0, j].set_title(f"#{sid} target ch{k}", fontsize=9)
        ax[1, j].imshow(pred[j, k], cmap="viridis")
        ax[1, j].set_title("prediction (short training)", fontsize=9)
    fig.suptitle("smoke test: tiny training subset, not a trained model", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
