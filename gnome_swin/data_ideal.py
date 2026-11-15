"""Dane do testu na IDEALNYCH sygnałach (bg == 'ideal', bez szumu): wejście z pików + cel z datasets/stacks.h5.

Wejście: tylko piki z ostatnich 120 s (z takiego okna powstaje akumulator), rozciągnięte na nt=2048 binów.
Cel: akumulator [256, 256] = maksimum po 37 kanałach magnitudy (wartość w stacks.h5 to uint8, dzielimy przez 8),
     dodatkowo dzielony przez NORM, żeby wartości były rzędu 0..1.
Metryka: położenie maksimum predykcji vs prawdziwy kierunek ściany (kąt po złożeniu półkul).
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1] / "datasets"
sys.path.insert(0, str(ROOT / "code"))
import dh_core as dh  # noqa: E402  (funkcje z dostarczonego zbioru: geometria obrazu 256x256)

from gnome_swin.loader import NT, sample_to_tensor  # noqa: E402

LIFETIME_S = 120.0
NORM = 15.0  # maksimum akumulatora ~ liczba stacji (15)
VALID = ~np.isnan(dh.pixel_directions()[..., 0])  # piksele wewnątrz koła


def ideal_ids(split):
    """sample_id idealnych ścian danego splitu, bez próbek 'don't care' (<6 pików w oknie lub magnituda <5 pT)."""
    m = pd.read_csv(ROOT / "meta.csv")
    m = m[(m.bg == "ideal") & (m.kind == "wall") & (m.split == split) & (m.n_in_window >= 6) & (m.magnitude >= 5)]
    return m.sample_id.tolist()


def true_channel(magnitude):
    """Kanał akumulatora odpowiadający prawdziwej magnitudzie: M_k = 5 * 1.25^k pT (najbliższy, k = 0..36)."""
    return int(np.clip(np.round(np.log(magnitude / 5.0) / np.log(1.25)), 0, 36))


def load_xy(df, sample_ids, sensors, target="max", nt=NT):
    """-> x [N, 3, nt, S] (ostatnie 120 s pików), y [N, 1, 256, 256], meta (DataFrame z kierunkami).
    target = "max": maksimum po 37 kanałach; "channel": jeden kanał, ten z prawdziwą magnitudą ściany."""
    x = torch.stack([sample_to_tensor(df, i, sensors, nt, LIFETIME_S) for i in sample_ids])  # okno 120 s w loaderze
    meta = pd.read_csv(ROOT / "meta.csv").set_index("sample_id").loc[sample_ids]
    with h5py.File(ROOT / "stacks.h5", "r") as f:
        if target == "channel":
            y = np.stack([f["x"][i, true_channel(m)].astype(np.float32) / 8.0 for i, m in zip(sample_ids, meta.magnitude)])
        else:
            y = np.stack([f["x"][i].astype(np.float32).max(0) / 8.0 for i in sample_ids])
    return x, torch.from_numpy(y / NORM)[:, None], meta


def angle_errors(pred, meta):
    """Dla każdej próbki: kąt [deg] między kierunkiem piksela maksimum predykcji a prawdziwym (oba złożone do z<=0),
    oraz wysokość maksimum predykcji. pred: [N, 1, 256, 256] (numpy/tensor)."""
    p = np.asarray(pred)[:, 0].copy()
    p[:, ~VALID] = -np.inf  # maksimum tylko wewnątrz koła
    dirs = dh.pixel_directions()
    errs, heights = [], []
    for k in range(len(p)):
        iy, ix = np.unravel_index(p[k].argmax(), p[k].shape)
        true = meta.iloc[k][["fold_x", "fold_y", "fold_z"]].values.astype(float)
        errs.append(float(dh.folded_angle_deg(dirs[iy, ix], true)))
        heights.append(float(p[k][iy, ix]))
    return np.array(errs), np.array(heights)
