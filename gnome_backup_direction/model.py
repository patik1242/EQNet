"""GnomeEQNet: the repository's real `EQNet` class with one extra head.

Reused unchanged (inherited from eqnet/models/eqnet.py):
  - EQNet.backbone      ResNet1D (resnet18): conv1 -> maxpool -> layer1..4 -> conv2, run per station
  - EQNet.phase_picker  PhasePicker  (original P/S/noise predictions, still callable)
  - EQNet.event_detector EventDetector (original event-detection output, still callable)
  - log_transform       signed log(1+|x|), applied to the pT amplitudes
New (the only added layers):
  - hough_head: Conv2d(128->64, 3x3) + BatchNorm2d + ReLU + Conv2d(64->1, 1x1), then bilinear resize to 256x256

Tensor shapes (B = batch, S = 15 stations, nt = 1024):
  data                 [B, 3, nt, S]
  backbone["out"]      [B, S, 128, nt/32]
  hough_head output    [B, 1, 256, 256]

The weights are random and nothing is trained: the output is an architecture test only.
Its two image axes are (time, station) resized to 256x256, NOT wall direction.
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
        x = x.permute(0, 2, 3, 1)  # [B, C, T, S]  (the only place where stations are mixed: the 3x3 kernel)
        x = self.layers(x)  # [B, 1, T, S]
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)  # [B, 1, 256, 256]


class HoughDenseHead(nn.Module):
    """Learnable (sensor, time) -> direction-image map; replaces the plain resize.

    Per-station EQNet features [B, S, C, T] -> 1x1 conv (C->16) -> max over time (T->8) -> flatten (S*16*8)
    -> Linear -> [32, 16, 16] -> 4x ConvTranspose2d (x2) -> 256x256 -> 1x1 conv -> n_out channels.
    The dense layer is what lets a given station/time pattern land on a *direction* pixel: the image axes are
    no longer tied to the input axes. Station identity = position in the flattened vector (fixed sensor order).
    """

    def __init__(self, in_channels=128, n_stations=15, n_out=37, t_pool=8, mid=16, base=32):
        super().__init__()
        self.t_pool, self.base = t_pool, base
        self.reduce = nn.Sequential(nn.Conv1d(in_channels, mid, 1, bias=False), nn.BatchNorm1d(mid), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(n_stations * mid * t_pool, base * 16 * 16), nn.ReLU())

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())

        self.decoder = nn.Sequential(up(base, 32), up(32, 32), up(32, 16), up(16, 16), nn.Conv2d(16, n_out, 1))

    def forward(self, features):
        x = features["out"]  # [B, S, C, T]
        b, s, c, t = x.shape
        x = self.reduce(x.reshape(b * s, c, t))  # [B*S, mid, T]
        x = F.adaptive_max_pool1d(x, self.t_pool)  # [B*S, mid, t_pool]
        x = self.fc(x.reshape(b, -1)).view(b, self.base, 16, 16)
        return self.decoder(x)  # [B, n_out, 256, 256]


class HoughGeoHead(nn.Module):
    """Geometry-aware head: one SHARED network for all sensors, sum over sensors (like arcs summed in an accumulator).

    Per sensor i the input is [EQNet features of that station (128 -> 16, max over time),
    strongest signed log-amplitude of the last 120 s, peak present, a_x, a_y, a_z, RF]; a shared MLP turns it into a
    [base, 16, 16] latent map, the maps of all sensors are summed, and a ConvTranspose decoder gives [B, n_out, 256, 256].
    Sensor identity enters only through its axis and RF (from geometry.py), not through its position, so the head
    has ~15x fewer parameters than HoughDenseHead and cannot memorise "column k = sensor k".
    """

    def __init__(self, geometry, in_channels=128, n_out=1, mid=16, hidden=256, base=32, recent=0.8):
        super().__init__()
        self.register_buffer("geom", torch.as_tensor(geometry, dtype=torch.float32))  # [S, 4]
        self.base, self.recent = base, recent  # recent: peaks after this fraction of the window (last 120 s of 600 s)
        self.reduce = nn.Sequential(nn.Conv1d(in_channels, mid, 1, bias=False), nn.BatchNorm1d(mid), nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(mid + 2 + 4, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, base * 16 * 16)
        )

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())

        self.decoder = nn.Sequential(up(base, 32), up(32, 32), up(32, 16), up(16, 16), nn.Conv2d(16, n_out, 1))

    def forward(self, features, data):
        x = features["out"]  # [B, S, C, T]
        b, s, c, t = x.shape
        f = F.adaptive_max_pool1d(self.reduce(x.reshape(b * s, c, t)), 1).view(b, s, -1)  # [B, S, mid]
        nt = data.shape[2]
        amp = data[:, 1, int(nt * self.recent) :, :]  # [B, nt', S] signed log A/RF, last 120 s
        c_i = torch.gather(amp, 1, amp.abs().argmax(1, keepdim=True)).squeeze(1)  # strongest peak, sign kept
        m_i = data[:, 2, int(nt * self.recent) :, :].amax(1)  # [B, S]
        z = torch.cat([f, c_i[..., None], m_i[..., None], self.geom.expand(b, -1, -1)], -1)  # [B, S, mid+2+4]
        z = self.mlp(z).view(b, s, self.base, 16, 16).sum(1)  # sum over sensors
        return self.decoder(z)


class GnomeEQNet(EQNet):
    def __init__(self, backbone="resnet18", size=(256, 256), head="dense", n_out=37, n_stations=15, geometry=None):
        super().__init__(backbone=backbone)
        self.head_type = head
        if head == "geo":  # sensor axes + RF (gnome/geometry.py), shared per-sensor network, sum over sensors
            self.hough_head = HoughGeoHead(geometry, 128, n_out)
        elif head == "dense":  # learnable mapping, n_out = 37 magnitude channels (like stacks.h5)
            self.hough_head = HoughDenseHead(128, n_stations, n_out)
        else:  # "resize": first PoC head, [B, 1, 256, 256], no learnable direction mapping
            self.hough_head = HoughHead(128, 64, size)

    def forward(self, batched_inputs, original_heads=False):
        data = batched_inputs["data"].to(self.device).clone()
        data[:, :2] = log_transform(data[:, :2])  # amplitudes span +-1e4 pT; sign is kept; mask channel untouched
        features = self.backbone(data)
        hough = self.hough_head(features, data) if self.head_type == "geo" else self.hough_head(features)
        out = {"hough": hough, "features": features["out"]}
        if original_heads:  # unchanged EQNet heads (eval mode returns (prediction, {}))
            out["phase"], _ = self.phase_picker(features)
            out["event"], _ = self.event_detector(features)
        return out
