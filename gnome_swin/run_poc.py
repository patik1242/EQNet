"""Test wersji wielostacyjnej (SwinTransformerV2) na pikach GNOME. Uruchomienie z katalogu głównego repo:
    python -m gnome_swin.run_poc [--sample_ids 0 1 2 3] [--out gnome_swin/first_image.png]
Bez treningu, losowe wagi. Dodatkowo test zależności między stacjami: zerujemy dane jednej stacji i sprawdzamy,
czy zmieniają się cechy INNYCH stacji (ResNet1D: nie, SwinTransformerV2: tak)."""
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from eqnet.models.eqnet import EQNet, log_transform
from gnome_swin.loader import get_sensors, load_signals, make_batch
from gnome_swin.model import GnomeEQNetSwin, station_locations


def cross_station_effect(features_fn, data, station=0):
    """Względna zmiana cech stacji innych niż `station` po wyzerowaniu danych tej stacji (0 = brak wpływu)."""
    with torch.no_grad():
        f0 = features_fn(data)
        d1 = data.clone()
        d1[..., station] = 0.0
        f1 = features_fn(d1)
        
    others = [i for i in range(f0.shape[1]) if i != station]
    return ((f1[:, others] - f0[:, others]).abs().mean() / f0[:, others].abs().mean()).item()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--sample_ids", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out", default="gnome_swin/first_image.png")
    a = p.parse_args()

    df = load_signals()
    sensors = get_sensors(df)
    batch = make_batch(df, a.sample_ids, sensors)
    print("wejście data:", tuple(batch["data"].shape), "n_peaks:", batch["n_peaks"])

    torch.manual_seed(0)
    swin = GnomeEQNetSwin(station_locations(sensors)).eval()

    with torch.no_grad():
        out = swin(batch, original_heads=True)

    print("cechy backbone (Swin):", tuple(out["features"].shape))
    print("oryginalne głowice: phase", tuple(out["phase"].shape), " event", tuple(out["event"].shape))
    print("wyjście hough:", tuple(out["hough"].shape))
    assert out["hough"].shape == (len(a.sample_ids), 1, 256, 256)

    # Czy pozycje stacji mają wpływ? Te same dane, pozycje w odwróconej kolejności.
    with torch.no_grad():
        other = swin(batch, locations=swin.locations.flip(0))["features"]
    print(f"zmiana cech po przestawieniu pozycji stacji: {(other - out['features']).abs().mean() / out['features'].abs().mean():.2e}")

    # Zależność między stacjami: ResNet1D (gnome/) vs SwinTransformerV2.
    resnet = EQNet(backbone="resnet18").eval()  # oryginalny EQNet z backbone ResNet1D

    def res_feat(d):
        x = d.clone()
        x[:, :2] = log_transform(x[:, :2])
        return resnet.backbone(x)["out"]

    swin_feat = lambda d: swin.features(d)["out"]
    print("\nWyzerowanie danych stacji 0 -> względna zmiana cech pozostałych 14 stacji:")
    print(f"  ResNet1D (oryginalny backbone): {cross_station_effect(res_feat, batch['data']):.2e}")
    print(f"  SwinTransformerV2 (okno 16 = 15 + 1 pusta): {cross_station_effect(swin_feat, batch['data']):.2e}")
    print("  (losowe wagi, inicjalizacja std=0.02: wpływ w Swin jest mały, ale niezerowy; ResNet1D nie łączy stacji)")

    n = len(a.sample_ids)
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)

    for i, sid in enumerate(a.sample_ids):
        ax[0, i].imshow(batch["data"][i, 1].numpy().T, aspect="auto", cmap="RdBu_r", vmin=-500, vmax=500)
        ax[0, i].set_title(f"#{sid} wall={batch['is_wall'][i]}", fontsize=9)
        ax[0, i].set_xlabel("bin czasu")
        ax[0, i].set_ylabel("sensor")
        ax[1, i].imshow(out["hough"][i, 0].numpy(), cmap="viridis")
        ax[1, i].set_title("wyjście bez treningu 256x256", fontsize=9)

    fig.suptitle("SWIN: TEST ARCHITEKTURY, losowe wagi, to NIE jest fizyczny akumulator Hougha", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("zapisano", a.out)


if __name__ == "__main__":
    main()
