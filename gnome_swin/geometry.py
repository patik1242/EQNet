"""Geometria magnetometrów z datasets/network_config_SR5.json: oś czułości i RF każdego sensora.

Config podaje oś lokalną (alt = wysokość nad horyzontem, az = azymut od północy ku wschodowi) oraz lat/lon.
Obracamy ją do układu związanego z Ziemią, w którym leży obraz Hougha:
    a = cos(alt) * (sin(az) * East + cos(az) * North) + sin(alt) * Up
(sprawdzone wcześniej na templates_bank.npz: szablon kąta theta = kierunki n z a·n = cos(theta)).
Obraz nie odróżnia z od -z, więc oś składamy do z >= 0.
Cecha sensora: [a_x, a_y, a_z, RF], RF = responseFactor (amplituda pika = M * (a·n) * RF).
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
    """[n_sensors, 4] = (a_x, a_y, a_z złożone do >= 0, RF) w kolejności `sensors`."""
    cfg = json.load(open(config))
    out = []
    for s in sensors:
        a = axis(cfg[s])
        a[2] = abs(a[2])
        out.append([*a, cfg[s]["responseFactor"]])
    return np.asarray(out, np.float32)
