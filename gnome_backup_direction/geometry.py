"""Sensor geometry from datasets/network_config_SR5.json.

The config gives each sensor a *local* axis (alt = elevation above the horizon, az = azimuth from north towards east)
plus lat/lon. Rotating it to the Earth-fixed frame in which the Hough image lives:
    a = cos(alt) * (sin(az) * East + cos(az) * North) + sin(alt) * Up
with East/North/Up built from lon/lat. Checked against templates_bank.npz (verify() below): for every sensor the
template of angle theta is the set of directions with a.n = cos(theta) (up to the image fold z -> -|z|).
The image cannot tell z from -z, so the axis is folded to z >= 0 (what the templates can resolve).
Feature vector per sensor: [a_x, a_y, a_z(folded), RF], RF = responseFactor (signed gain).
"""
import json
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[1] / "datasets" / "network_config_SR5.json"


def axis(cfg: dict) -> np.ndarray:
    lat, lon, alt, az = (np.radians(cfg[k]) for k in ("lat", "lon", "alt", "az"))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    a = np.cos(alt) * (np.sin(az) * east + np.cos(az) * north) + np.sin(alt) * up
    return a / np.linalg.norm(a)


def sensor_features(sensors, config=CONFIG) -> np.ndarray:
    """[n_sensors, 4] float32 = (a_x, a_y, a_z folded to >= 0, responseFactor) in the given sensor order."""
    cfg = json.load(open(config))
    out = []
    for s in sensors:
        a = axis(cfg[s])
        a[2] = abs(a[2])
        out.append([*a, cfg[s]["responseFactor"]])
    return np.asarray(out, np.float32)


def verify(templates_npz=None, tol_deg: float = 5.0) -> float:
    """Fraction of template pixels consistent with a.n = cos(theta) for the computed axes (1.0 = all). Needs dh_core."""
    import sys

    root = CONFIG.parent
    sys.path.insert(0, str(root / "code"))
    import dh_core as dh

    t = np.load(templates_npz or root / "templates_bank.npz")
    keys = [str(k) for k in t["sensor_keys"]]
    cfg = json.load(open(CONFIG))
    pd_ = dh.pixel_directions()[::4, ::4].reshape(-1, 3)
    ok = ~np.isnan(pd_[:, 0])
    pd_ = pd_[ok]
    mir = pd_ * np.array([1, 1, -1])
    worst = 1.0
    for i, k in enumerate(keys):
        s = t["templates"][i][:, ::4, ::4].reshape(37, -1)[:, ok].astype(float)
        cov = s.max(0) > 0
        th = np.radians(5.0 * s.argmax(0)[cov])
        a = axis(cfg[k])
        d = np.minimum(np.abs(np.arccos(np.clip(pd_[cov] @ a, -1, 1)) - th),
                       np.abs(np.arccos(np.clip(mir[cov] @ a, -1, 1)) - th))
        worst = min(worst, float((d < np.radians(tol_deg)).mean()))
    return worst


if __name__ == "__main__":
    print("worst-sensor agreement with templates_bank.npz:", round(verify(), 3))
