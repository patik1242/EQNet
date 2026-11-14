"""Forward-pass test of the adapted EQNet on GNOME peaks.  Run from the repo root:
    python -m gnome.run_poc [--sample_ids 0 1 2 3] [--out gnome/first_image.png]
No training. Random weights."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gnome.loader import GnomeSignals, describe
from gnome.model import GnomeEQNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--sample_ids", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out", default="gnome/first_image.png")
    a = p.parse_args()

    ds = GnomeSignals(a.csv) if a.csv else GnomeSignals()
    for k, v in describe(ds.df).items():
        print(f"{k}: {v}")

    torch.manual_seed(0)
    model = GnomeEQNet(head="resize").eval()  # first PoC (resize head); training: gnome/train_smoke.py
    batch = ds.batch(a.sample_ids)
    print("input  data:", tuple(batch["data"].shape), "n_peaks:", batch["n_peaks"])
    with torch.no_grad():
        out = model(batch, original_heads=True)
    print("backbone features:", tuple(out["features"].shape))
    print("original phase_picker:", tuple(out["phase"].shape), " event_detector:", tuple(out["event"].shape))
    print("hough output:", tuple(out["hough"].shape))
    assert out["hough"].shape == (len(a.sample_ids), 1, 256, 256)

    n = len(a.sample_ids)
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
    for i, sid in enumerate(a.sample_ids):
        ax[0, i].imshow(batch["data"][i, 1].numpy().T, aspect="auto", cmap="RdBu_r", vmin=-500, vmax=500)
        ax[0, i].set_title(f"#{sid} wall={batch['is_wall'][i]}", fontsize=9)
        ax[0, i].set_xlabel("time bin")
        ax[0, i].set_ylabel("sensor")
        ax[1, i].imshow(out["hough"][i, 0].numpy(), cmap="viridis")
        ax[1, i].set_title("untrained output 256x256", fontsize=9)
    fig.suptitle("TEST OF ARCHITECTURE: random weights, NOT a physical Hough accumulator", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
