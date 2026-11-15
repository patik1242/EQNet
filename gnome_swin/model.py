"""GnomeEQNetSwin: klasa `EQNet` z repozytorium, ale z backbone wielostacyjnym (SwinTransformerV2) zamiast ResNet1D.

Po co: ResNet1D liczy każdą stację osobno. SwinTransformerV2 (eqnet/models/swin_transformer_v2.py) ma uwagę w oknach
(czas x stacje) z "continuous relative position bias": cpb_mlp dostaje względne współrzędne (dt, dx, dy) tokenów,
gdzie (x, y) to położenie stacji. Dzięki temu stacje wpływają na siebie nawzajem (to mechanizm z artykułu o EQNet2).

Użyte z repozytorium bez zmian: EQNet (oryginalne głowice phase_picker i event_detector), SwinTransformerV2, log_transform.
Różnice względem `EQNet(backbone="swin2")`: okno uwagi obejmuje wszystkie stacje (16 = 15 magnetometrów + 1 pusta),
a nie [7, 10] zaszyte w EQNet. Pusta stacja sprawia, że oś stacji jest parzysta i mieści się w jednym oknie.
Loader sygnałów jest w gnome_swin/loader.py, a głowica HoughHead w tym pliku.

Kształty (B = batch, S = 15 stacji, nt = 2048):
  data [B, 3, nt, S] (+1 pusta kolumna), station_location [B, S, 2] (+1 zerowa pozycja)
  -> backbone["out"] [B, 16, 128, nt/32] -> obcięcie do [B, S, 128, nt/32] -> HoughHead -> [B, 1, 256, 256]
Wagi losowe, bez treningu.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eqnet.models.eqnet import EQNet, log_transform
from eqnet.models.swin_transformer_v2 import SwinTransformerV2

CONFIG = Path(__file__).resolve().parents[1] / "datasets" / "network_config_SR5.json"


def station_locations(sensors, config=CONFIG) -> torch.Tensor:
    """[S, 2] = (lon, lat) / 180 w kolejności sensors. Prosta normalizacja, nie jest to R/theta z artykułu."""
    cfg = json.load(open(config))
    return torch.tensor([[cfg[s]["lon"] / 180.0, cfg[s]["lat"] / 180.0] for s in sensors], dtype=torch.float32)


def add_geometry(data: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    """Dokleja geometrię sensorów jako stałe w czasie kanały: data [B, 3, nt, S] + geometry [S, 4] -> [B, 7, nt, S].
    Kanały geometrii: oś czułości (a_x, a_y, a_z) i RF danej stacji (ta sama wartość w każdym kroku czasu)."""
    B, _, nt, S = data.shape
    g = geometry.to(data.device).t()[None, :, None, :].expand(B, -1, nt, S)
    return torch.cat([data, g], 1)


class HoughHead(nn.Module):
    """Jedyne nowe warstwy poza backbone: Conv2d(128->64, 3x3) + norma + ReLU + Conv2d(64->1, 1x1), skalowanie do 256x256.
    Norma: GroupNorm (wynik nie zależy od innych próbek w batchu, więc train() i eval() liczą to samo) albo BatchNorm."""

    def __init__(self, in_channels: int = 128, hidden: int = 64, size=(256, 256), norm: str = "group"):
        super().__init__()
        self.size = size
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden) if norm == "group" else nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, features):
        x = features["out"]  # [B, S, C, T]
        x = x.permute(0, 2, 3, 1)  # [B, C, T, S]  (jądro 3x3 miesza też sąsiednie kolumny stacji)
        x = self.layers(x)  # [B, 1, T, S]
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)  # [B, 1, 256, 256]


class GnomeEQNetResNet(nn.Module):
    """Do porównania: backbone ResNet1D z EQNet (stacje liczone osobno) + HoughHead.
    geometry [S, 4] (opcjonalnie): osie i RF sensorów jako dodatkowe kanały wejścia (pierwsza Conv1d: 3 -> 3+4 kanałów)."""

    def __init__(self, geometry=None, size=(256, 256), norm: str = "group"):
        super().__init__()
        self.backbone = EQNet(backbone="resnet18").backbone
        self.register_buffer("geometry", geometry)
        if geometry is not None:  # oryginalna Conv1d(3, 32, 7) ma na sztywno 3 kanały
            self.backbone.conv1 = nn.Conv1d(3 + geometry.shape[1], 32, 7, 2, 3, bias=False, padding_mode="reflect")
            nn.init.kaiming_normal_(self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.hough_head = HoughHead(128, 64, size, norm)

    def features(self, data):
        """[B, 3, nt, S] -> {"out": [B, S, 128, nt/32]}"""
        x = data.clone()
        x[:, :2] = log_transform(x[:, :2])
        if self.geometry is not None:
            x = add_geometry(x, self.geometry)
        return self.backbone(x)

    def forward(self, data):
        return self.hough_head(self.features(data))


class GnomeEQNetSwin(EQNet):
    def __init__(self, locations: torch.Tensor, size=(256, 256), stochastic_depth: float = 0.2, geometry=None,
                 norm: str = "group", window_t: int = 7):
        super().__init__(backbone="swin2")  # oryginalny EQNet; backbone podmieniamy niżej
        self.n_stations = len(locations)
        # PatchMerging dopełnia oś stacji do liczby parzystej. Gdy okno uwagi jest mniejsze od tej osi, okna się dzielą
        # i przesuwają względem pozycji stacji (sprawdzone: sieć przestaje zależeć tylko od geometrii). Dlatego dokładamy
        # jedną pustą stację (zera) i ustawiamy okno = parzysta liczba stacji; wtedy wszystkie stacje są w jednym oknie.
        self.n_tokens = self.n_stations + self.n_stations % 2
        self.backbone = SwinTransformerV2(  # te same hiperparametry co w EQNet(backbone="swin2"), inne okno
            patch_size=[4, 1], embed_dim=16, depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 8],
            window_size=[window_t, self.n_tokens],  # okno w czasie (domyślnie 7 jak w EQNet swin2) x wszystkie stacje stochastic_depth_prob=stochastic_depth,
        )
        self.register_buffer("locations", locations)  # [S, 2]
        self.register_buffer("geometry", geometry)  # [S, 4] albo None
        if geometry is not None:  # pierwsza warstwa Swin (Conv2d 3 -> embed_dim) ma na sztywno 3 kanały
            self.backbone.split_patch[0] = nn.Conv2d(3 + geometry.shape[1], 16, kernel_size=(4, 1), stride=(4, 1))
        self.hough_head = HoughHead(128, 64, size, norm)

    def features(self, data, locations=None):
        """Backbone na wejściu [B, 3, nt, S] -> cechy [B, S, 128, nt/32] (pustą dodatkową stację odcinamy)."""
        data = data.to(self.device).clone()
        data[:, :2] = log_transform(data[:, :2])  # amplitudy +-1e4 pT, znak zachowany
        if self.geometry is not None:
            data = add_geometry(data, self.geometry)
        loc = self.locations if locations is None else locations  # [S, 2]
        n_pad = self.n_tokens - self.n_stations
        if n_pad:  # pusta stacja: zera w danych (też w geometrii) i w pozycji
            data = torch.cat([data, data.new_zeros(*data.shape[:-1], n_pad)], -1)
            loc = torch.cat([loc, loc.new_zeros(n_pad, 2)], 0)
        out = self.backbone(data, loc.expand(data.shape[0], -1, -1))["out"]
        return {"out": out[:, : self.n_stations].contiguous()}  # contiguous: oryginalne głowice używają .view

    def forward(self, batched_inputs, original_heads=False, locations=None):
        features = self.features(batched_inputs["data"], locations)
        out = {"hough": self.hough_head(features), "features": features["out"]}
        if original_heads:  # oryginalne głowice EQNet bez zmian
            out["phase"], _ = self.phase_picker(features)
            out["event"], _ = self.event_detector(features)
        return out
