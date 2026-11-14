# GNOME x EQNet: test adaptacji (bez treningu)

Tylko `datasets/signals.csv`. Oryginalne pliki repozytorium i dane nie są zmieniane (nowy katalog `gnome/`).

Uruchomienie z katalogu głównego repo (wymaga `torch`, `pandas`, `matplotlib`, `obspy`):

    python -m gnome.run_poc --sample_ids 0 1 2 3 --out gnome/first_image.png

## Pliki
- `loader.py` – funkcje `load_signals`, `get_sensors`, `sample_to_tensor`, `make_batch`: CSV -> tensor `[B, 3, 2048, 15]` (kanały: amplitude_pT, amplitude_corr_pT, maska piku; znaki zachowane).
- `model.py` – `GnomeEQNet(EQNet)` + `HoughHead`.
- `run_poc.py` – forward pass i wizualizacja.

## Co oryginalne, co nowe
- Oryginalne, bez zmian: `eqnet.models.eqnet.EQNet` (`backbone` = ResNet1D resnet18, `phase_picker` = PhasePicker, `event_detector` = EventDetector), `log_transform`.
- Nowe: `HoughHead` (Conv2d 128->64 3x3, BatchNorm2d, ReLU, Conv2d 64->1 1x1, interpolacja bilinearna do 256x256) i `GnomeEQNet.forward`.

## Kształty
`data [B,3,2048,15]` -> backbone `[B,15,128,64]` -> `[B,1,256,256]`.
Oryginalne głowice nadal działają: `phase [B,3,2048,15]`, `event [B,128,15]`.

## Ograniczenia
- Losowe wagi; obraz to test architektury, nie akumulator Hougha. Osie obrazu to (czas, sensor) przeskalowane do 256x256, a nie kierunek ściany.
- Backbone liczy każdą stację osobno; sensory miesza dopiero 3x3 w `HoughHead` (kolejność sensorów jest alfabetyczna, bez znaczenia fizycznego).
- Zapisujemy 1 kanał; docelowo 37 kanałów magnitud i etykiety z `stacks.h5`.
- EQNet2 / `network_config` (położenia stacji, `swin2`) nie jest używane.
- `nt >= 2048`, bo oryginalny EventDetector wymaga >= 64 kroków po backbone.

## Dalsze kroki
Dodać etykiety (akumulatory z `stacks.h5`), zastąpić resize głowicą uczoną (ConvTranspose/dekoder), użyć pozycji stacji, dopiero potem trenować.
