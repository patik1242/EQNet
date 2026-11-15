"""Test: czy EQNet (ResNet1D albo Swin) potrafi odwzorować akumulator Hougha na IDEALNYCH danych?
Uruchomienie z katalogu głównego repo:
    python -m gnome_swin.test_ideal --backbone swin --mode overfit --n 8 --target channel --geometry
    python -m gnome_swin.test_ideal --backbone resnet --mode train --epochs 40
Tryby: overfit = uczenie i ocena na tych samych n próbkach train (czy w ogóle da się odwzorować),
       train   = uczenie na train, ocena na val+test (czy uogólnia).
Cel: --target max (maksimum po 37 kanałach) albo channel (jeden kanał: ten z prawdziwą magnitudą ściany).
Miary: błąd kąta maksimum predykcji [deg], korelacja predykcji z celem, MSE; MSE w trybie train() vs eval()
(kontrola BatchNorm; z GroupNorm w głowicy różnica pochodzi tylko z BatchNorm w ResNet1D)."""
import argparse
import copy
import sys
import time

import numpy as np
import torch

from gnome_swin.data_ideal import VALID, angle_errors, ideal_ids, load_xy
from gnome_swin.geometry import sensor_features
from gnome_swin.loader import get_sensors, load_signals
from gnome_swin.model import GnomeEQNetResNet, GnomeEQNetSwin, station_locations


class SwinHough(torch.nn.Module):
    """Opakowanie, żeby oba modele miały to samo wywołanie model(data) -> [B, 1, 256, 256]."""

    def __init__(self, sensors, sd, geometry, norm, window_t=7):
        super().__init__()
        self.m = GnomeEQNetSwin(station_locations(sensors), stochastic_depth=sd, geometry=geometry, norm=norm,
                                window_t=window_t)

    def features(self, data):
        return self.m.features(data)

    def forward(self, data):
        return self.m({"data": data})["hough"]


def station_dependence(model, x, stations=(0, 4, 9, 14)):
    """Zależność cech jednej stacji od sygnałów pozostałych: zerujemy 3 kanały sygnału stacji k (geometria zostaje)
    i liczymy względną zmianę cech wszystkich INNYCH stacji; średnia po k. 0 = stacje się nie widzą (ResNet1D).
    Dodatkowo: rozrzut cech między próbkami (std po próbkach / std całkowite): 0 = cechy nie zależą od wejścia."""
    model.eval()
    with torch.no_grad():
        f0 = model.features(x)["out"]

        eff = []
        for k in stations:
            d = x.clone()
            d[:, :, :, k] = 0.0
            f1 = model.features(d)["out"]
            others = [i for i in range(f0.shape[1]) if i != k]
            eff.append(((f1[:, others] - f0[:, others]).abs().mean() / f0[:, others].abs().mean()).item())
        between = (f0.std(0).mean() / f0.std()).item()

    return float(np.mean(eff)), between


def predict(model, x, bs=8):
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])


def evaluate(model, x, y, meta):
    """MSE, korelacja i błąd kąta w trybie eval() oraz MSE w trybie train() (kopia modelu, żeby nie ruszać statystyk BN)."""
    model.eval()
    pred = predict(model, x)
    train_mode = copy.deepcopy(model).train()
    mse_train_mode = ((predict(train_mode, x) - y)[:, :, VALID] ** 2).mean().item()
    err, h = angle_errors(pred.numpy(), meta)
    mse = ((pred - y)[:, :, VALID] ** 2).mean().item()
    p, t = pred[:, 0][:, VALID], y[:, 0][:, VALID]  # [N, liczba pikseli w kole]
    p, t = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    corr = ((p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1) + 1e-8)).mean().item()  # korelacja per próbka, uśredniona
    return err, h, mse, mse_train_mode, corr


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()

    p.add_argument("--backbone", choices=["resnet", "swin"], default="swin")
    p.add_argument("--mode", choices=["overfit", "train"], default="overfit")
    p.add_argument("--target", choices=["max", "channel"], default="max")
    p.add_argument("--n", type=int, default=8, help="overfit: ile próbek")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--sd", type=float, default=0.0, help="stochastic depth Swin (0 = wyłączone)")
    p.add_argument("--clip", type=float, default=0.0, help="przycinanie normy gradientu (0 = brak)")
    p.add_argument("--geometry", action="store_true", help="dodaj osie czułości i RF sensorów do wejścia")
    p.add_argument("--norm", choices=["group", "batch"], default="group", help="normalizacja w głowicy")
    p.add_argument("--nt", type=int, default=2048, help="liczba binów czasu na 120 s (reprezentacja czasowa wejścia)")
    p.add_argument("--win_t", type=int, default=7, help="Swin: okno attention w czasie (tokeny)")
    p.add_argument("--threads", type=int, default=0, help="liczba wątków torch (0 = domyślnie)")
    
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    df = load_signals()
    sensors = get_sensors(df)
    geometry = torch.from_numpy(sensor_features(sensors)) if a.geometry else None
    tr_ids = ideal_ids("train")

    if a.mode == "overfit":
        tr_ids = tr_ids[: a.n]
        ev = {"train(te same)": load_xy(df, tr_ids, sensors, a.target, a.nt)}
    else:
        ev = {"val": load_xy(df, ideal_ids("val"), sensors, a.target, a.nt), "test": load_xy(df, ideal_ids("test"), sensors, a.target, a.nt)}
    
    x, y, meta = load_xy(df, tr_ids, sensors, a.target, a.nt)
    print(f"{vars(a)}\npróbek train={len(x)} eval={ {k: len(v[0]) for k, v in ev.items()} }")

    # punkt odniesienia: stała mapa = średni cel z train (nie zależy od wejścia)
    mean_map = y.mean(0, keepdim=True)
    for k, (xe, ye, me) in ev.items():
        e, _ = angle_errors(mean_map.expand(len(xe), -1, -1, -1).numpy(), me)
        print(f"  baseline 'średnia mapa' {k}: mediana błędu kąta {np.median(e):.1f} deg, MSE {((mean_map - ye)[:, :, VALID] ** 2).mean():.4f}")

    model = GnomeEQNetResNet(geometry, norm=a.norm) if a.backbone == "resnet" else SwinHough(sensors, a.sd, geometry, a.norm, a.win_t)
    print("parametrów:", f"{sum(q.numel() for q in model.parameters()):,}")
    dep0, btw0 = station_dependence(model, x)
    print(f"przed treningiem: zależność od innych stacji {dep0:.2e}, rozrzut cech między próbkami {btw0:.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    mask = torch.from_numpy(VALID)

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(x))
        tot = 0.0
        for i in range(0, len(x), a.bs):
            idx = perm[i:i + a.bs]
            
            if len(idx) < 2:  # BatchNorm w ResNet1D potrzebuje >1 próbki
                continue
            loss = ((model(x[idx]) - y[idx]) ** 2)[:, :, mask].mean()
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else float("inf"))
            opt.step()
            tot += loss.item() * len(idx)
        
        
        if ep % a.eval_every == 0 or ep == a.epochs:
            msg = f"ep {ep:3d} loss {tot / len(x):.4f} norma_grad {gn:.2f} ({time.time() - t0:.0f}s)"
            for k, (xe, ye, me) in ev.items():
                e, h, mse, mse_tr, corr = evaluate(model, xe, ye, me)
                msg += (f" | {k}: kąt med {np.median(e):.1f} deg, <15deg {np.mean(e < 15):.2f}, korelacja {corr:.2f}, "
                        f"MSE eval {mse:.4f} / train-mode {mse_tr:.4f}")
            print(msg, flush=True)
    
    dep, btw = station_dependence(model, x)
    print(f"po treningu: zależność od innych stacji {dep:.2e} (przed: {dep0:.2e}), rozrzut cech między próbkami {btw:.3f} (przed: {btw0:.3f})")


if __name__ == "__main__":
    main()
