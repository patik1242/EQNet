"""Minimal GNOME loader: datasets/signals.csv -> EQNet-compatible tensor.

EQNet's backbone (eqnet/models/resnet1d.py) expects  data: [batch, channel=3, time, station].
Each sample here is a list of peaks (sensor, t_rel_s, amplitude), so we rasterize peaks onto a regular
time grid, one column per magnetometer. Signs of the amplitudes are preserved.

Channels of the produced tensor [3, nt, n_sensors]:
  0: amplitude_pT       (signed, raw, with RF)
  1: amplitude_corr_pT  (signed, A/RF - the value hough4gnome uses)
  2: peak mask          (1 where a peak exists)
Empty bins are 0, so a different number of peaks per sample/sensor needs no special handling.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "datasets" / "signals.csv"
WINDOW_S = 600.0  # signals.csv holds peaks in (-600, 0] s before T


def load_signals(path=DEFAULT_CSV) -> pd.DataFrame:
    """Read signals.csv (never modified)."""
    return pd.read_csv(path)


def describe(df: pd.DataFrame) -> dict:
    """Real structure of the file."""
    per_sample = df.groupby("sample_id").size()
    return {
        "columns": df.columns.tolist(),
        "n_rows": len(df),
        "n_samples": df.sample_id.nunique(),
        "sample_id_range": (int(df.sample_id.min()), int(df.sample_id.max())),
        "sensors": sorted(df.sensor.unique()),
        "t_rel_s_range": (float(df.t_rel_s.min()), float(df.t_rel_s.max())),
        "peaks_per_sample": (int(per_sample.min()), float(per_sample.mean()), int(per_sample.max())),
        "max_peaks_per_sensor_in_sample": int(df.groupby(["sample_id", "sensor"]).size().max()),
        "sources": df.source.value_counts().to_dict(),
    }


DEFAULT_STACKS = DEFAULT_CSV.parent / "stacks.h5"
DEFAULT_META = DEFAULT_CSV.parent / "meta.csv"
STORE_SCALE = 8.0  # accumulator value = uint8 / 8 (README; the h5 file carries no attributes)


def valid_mask(size: int = 256) -> torch.Tensor:
    """Pixels inside the projection disk (r <= 128 px); corners carry no information."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return (((xx - size / 2) ** 2 + (yy - size / 2) ** 2).sqrt() <= size / 2).float()


class GnomeSignals:
    """Signals (+ optional reference accumulators from stacks.h5 and labels from meta.csv)."""

    def __init__(self, path=DEFAULT_CSV, nt: int = 2048, window_s: float = WINDOW_S, stacks=None, meta=None,
                 with_stacks: bool = True):
        self.df = load_signals(path)
        self.with_stacks = with_stacks  # False: skip reading the accumulators (direction task needs only meta.csv)
        # optional files: used only if they exist; the signals-only path keeps working without them
        stacks = DEFAULT_STACKS if stacks is None else stacks
        meta = DEFAULT_META if meta is None else meta
        self.meta = pd.read_csv(meta).set_index("sample_id") if Path(meta).exists() else None
        self._stacks_path = Path(stacks) if Path(stacks).exists() else None
        self._h5 = None
        # ResNet1D downsamples time by 32 and the original EventDetector (dilation 16, reflect pad 32)
        # needs >= 64 steps after the backbone -> nt >= 2048 (0.29 s per bin for 600 s)
        self.nt = nt
        self.window_s = window_s
        self.sensors = sorted(self.df.sensor.unique())  # fixed station order -> station axis
        self.sensor_idx = {s: i for i, s in enumerate(self.sensors)}
        self.groups = {sid: g for sid, g in self.df.groupby("sample_id")}
        self.sample_ids = sorted(self.groups)

    def __len__(self):
        return len(self.sample_ids)

    def get(self, sample_id: int) -> dict:
        g = self.groups[sample_id]
        x = np.zeros((3, self.nt, len(self.sensors)), dtype=np.float32)
        # t_rel in (-window, 0]  ->  bin 0..nt-1
        it = np.clip(((g.t_rel_s.values + self.window_s) / self.window_s * self.nt).astype(int), 0, self.nt - 1)
        for k, (s, a_raw, a_corr) in enumerate(zip(g.sensor.values, g.amplitude_pT.values, g.amplitude_corr_pT.values)):
            i = self.sensor_idx[s]
            if x[2, it[k], i] == 1 and abs(a_corr) <= abs(x[1, it[k], i]):
                continue  # two peaks in one bin: keep the larger one
            x[0, it[k], i], x[1, it[k], i], x[2, it[k], i] = a_raw, a_corr, 1.0
        item = {
            "data": torch.from_numpy(x),  # [3, nt, n_sensors]
            "sample_id": int(sample_id),
            "n_peaks": len(g),
            "is_wall": bool(g.is_wall.max()),
        }
        if self._stacks_path is not None and self.with_stacks:
            item["target"] = self.target(sample_id)
        if self.meta is not None:
            item["heatmap"] = self.heatmap(sample_id)
            item["dont_care"] = self.dont_care(sample_id)
        return item

    # --- direction task (needs only meta.csv + datasets/code/dh_core.py) ---------------------------
    def dont_care(self, sample_id, min_in_window: int = 6, min_magnitude: float = 5.0, false_as="dont_care") -> bool:
        """Same rule as datasets/code/loader.py: weak/short walls (and FALSE) must not be scored either way."""
        r = self.meta.loc[sample_id]
        if r.kind == "false":
            return false_as == "dont_care"
        return bool(r.has_wall) and not (r.n_in_window >= min_in_window and r.magnitude >= min_magnitude)

    def heatmap(self, sample_id, sigma_deg: float = 2.0) -> torch.Tensor:
        """[1, 256, 256] Gaussian on the sphere around the folded wall direction (zeros: no wall / don't care)."""
        import sys

        sys.path.insert(0, str(DEFAULT_CSV.parent / "code"))
        import dh_core as dh

        r = self.meta.loc[sample_id]
        wall = bool(r.has_wall) and not self.dont_care(sample_id) and r.kind != "false"
        hm = dh.heatmap([[r.dir_x, r.dir_y, r.dir_z]] if wall else [], sigma_deg)
        return torch.from_numpy(hm)[None]

    def target(self, sample_id: int) -> torch.Tensor:
        """Reference accumulator [37, 256, 256] (float, = uint8 / 8). sample_id is the row index in stacks.h5."""
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self._stacks_path, "r")
        return torch.from_numpy(self._h5["x"][sample_id].astype(np.float32) / STORE_SCALE)

    def ids(self, split: str = None, kinds=None):
        """sample_ids filtered by meta.csv `split` (train/val/test) and `kind`."""
        m = self.meta
        keep = m.index.isin(self.sample_ids)
        if split is not None:
            keep &= (m.split == split).values
        if kinds is not None:
            keep &= m.kind.isin(kinds).values
        return [int(i) for i in m.index[keep]]

    def batch(self, sample_ids) -> dict:
        items = [self.get(s) for s in sample_ids]
        out = {
            "data": torch.stack([i["data"] for i in items]),  # [B, 3, nt, n_sensors]
            "sample_id": [i["sample_id"] for i in items],
            "n_peaks": [i["n_peaks"] for i in items],
            "is_wall": [i["is_wall"] for i in items],
        }
        if "target" in items[0]:
            out["target"] = torch.stack([i["target"] for i in items])  # [B, 37, 256, 256]
        if "heatmap" in items[0]:
            out["heatmap"] = torch.stack([i["heatmap"] for i in items])  # [B, 1, 256, 256]
            out["dont_care"] = torch.tensor([i["dont_care"] for i in items])  # [B] bool
        return out
