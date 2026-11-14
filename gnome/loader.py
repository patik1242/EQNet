"""Loader GNOME: datasets/signals.csv → tensor dla EQNet. Zwykłe funkcje, bez klas.

Backbone EQNet (eqnet/models/resnet1d.py) 
oczekuje danych w kształcie [batch, 3 kanały, czas, stacje]. 
Próbka GNOME to lista pików (sensor, t_rel_s, amplituda), więc układamy piki na regularnej siatce czasu, 
po jednej kolumnie na magnetometr. Znaki amplitud są zachowane. Puste biny to 0, więc działa dowolna liczba pików.

"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "datasets" / "signals.csv"
WINDOW_S = 600.0  #signals.csv zawiera piki z 600 s 
NT = 2048

def load_signals(path=DEFAULT_CSV):
    return pd.read_csv(path)

#sieć uczy się po pozycji kolumny dlatego kolejność alfabetyczna
def get_sensors(df):
    return sorted(df.sensor.unique())

#sprawdzenie co jest w pliku: jaki zakres sample_id, sensory, kolumny
def describe(df):
    per_sample = df.groupby("sample_id").size()

    return {
        "columns": df.columns.tolist(),
        "n_rows": len(df),
        "n_samples": df.sample_id.nunique(),
        "sample_id_range": (int(df.sample_id.min()), int(df.sample_id.max())),
        "sensors": get_sensors(df),
        "t_rel_s_range": (float(df.t_rel_s.min()), float(df.t_rel_s.max())),
        "peaks_per_sample": (int(per_sample.min()), float(per_sample.mean()), int(per_sample.max())),
        "max_peaks_per_sensor_in_sample": int(df.groupby(["sample_id", "sensor"]).size().max()),
        "sources": df.source.value_counts().to_dict(),
    }


#zamiana listy pików jednej próbki (sensor, czas, aplituda) na to co wymaga EQNet [3, nt, n_sensors]
#nt -> liczba korków czasu, 3 bo pierwsza warstwa backbone EQNet to na sztywno Conv1D(3,..) więc 
#wejście musi mieć 3 kanały 
def sample_to_tensor(df: pd.DataFrame, sample_id: int, sensors: list, nt: int = NT, window_s: float = WINDOW_S):
    peaks = df[df.sample_id == sample_id]
    x = np.zeros((3, nt, len(sensors)), dtype=np.float32)

    bins = np.clip(((peaks.t_rel_s.values + window_s) / window_s * nt).astype(int), 0, nt - 1)
    cols = [sensors.index(s) for s in peaks.sensor.values]  # sensor -> kolumna

    for t, c, a_raw, a_corr in zip(bins, cols, peaks.amplitude_pT.values, peaks.amplitude_corr_pT.values):
        if x[2, t, c] == 1 and abs(a_corr) <= abs(x[1, t, c]):
            continue  

        x[0, t, c], x[1, t, c], x[2, t, c] = a_raw, a_corr, 1.0
    return torch.from_numpy(x)


#składa kilka tensorów w jeden batch i dodaje informacje pomocnicze takie jak n_peaks, is_wall
def make_batch(df: pd.DataFrame, sample_ids, sensors: list, nt: int = NT) -> dict:
    data, n_peaks, is_wall = [], [], []

    for sid in sample_ids:
        peaks = df[df.sample_id == sid]
        data.append(sample_to_tensor(df, sid, sensors, nt))
        n_peaks.append(len(peaks))
        is_wall.append(bool(peaks.is_wall.max()))

    return {
        "data": torch.stack(data), 
        "sample_id": list(sample_ids), 
        "n_peaks": n_peaks, ""
        "is_wall": is_wall
    }
