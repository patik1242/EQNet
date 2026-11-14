"""GnomeEQNet: prawdziwa klasa `EQNet` z repozytorium plus jedna dodatkowa głowica.

Użyte bez zmian:
  - EQNet.backbone       ResNet1D (resnet18): conv1 -> maxpool -> layer1..4 -> conv2, liczony osobno dla każdej stacji
  - EQNet.phase_picker   PhasePicker  (oryginalne predykcje P/S/szum, nadal można wywołać)
  - EQNet.event_detector EventDetector (oryginalna detekcja zdarzeń, nadal można wywołać)
  - log_transform        logarytm ze znakiem log(1+|x|), stosowany do amplitud w pT
Nowe (jedyne dodane warstwy):
  - hough_head: Conv2d(128->64, 3x3) + BatchNorm2d + ReLU + Conv2d(64->1, 1x1), potem skalowanie bilinearne do 256x256

Kształty tensorów (B = batch, S = 15 stacji, nt = 2048):
  data                 [B, 3, nt, S]
  backbone["out"]      [B, S, 128, nt/32]
  wyjście hough_head   [B, 1, 256, 256]

Wagi są losowe i nic nie jest trenowane: wynik to tylko test architektury.
Osie obrazu to (czas, stacja) przeskalowane do 256x256, a NIE kierunek ściany.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from eqnet.models.eqnet import EQNet, log_transform


class HoughHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden: int = 64, size=(256, 256)):
        super().__init__()
        self.size = size
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, features):
        x = features["out"]  # [B, S, C, T]
        x = x.permute(0, 2, 3, 1)  # [B, C, T, S]  (jedyne miejsce mieszania stacji: jądro 3x3)
        x = self.layers(x)  # [B, 1, T, S]
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)  # [B, 1, 256, 256]


class GnomeEQNet(EQNet):
    def __init__(self, backbone="resnet18", size=(256, 256)):
        super().__init__(backbone=backbone)
        self.hough_head = HoughHead(128, 64, size)

    def forward(self, batched_inputs, original_heads=False):
        data = batched_inputs["data"].to(self.device).clone()
        data[:, :2] = log_transform(data[:, :2])  # amplitudy w zakresie +-1e4 pT; znak zachowany; maska bez zmian
        
        features = self.backbone(data)
        out = {"hough": self.hough_head(features), "features": features["out"]}

        if original_heads:  # oryginalne głowice EQNet bez zmian (w trybie eval zwracają (predykcja, {}))
            out["phase"], _ = self.phase_picker(features)
            out["event"], _ = self.event_detector(features)
        return out
