# GNOME x EQNet: adaptacja (prototyp)

Oryginalne pliki repozytorium i dane w `datasets/` nie są zmieniane (wszystko w `gnome/`).

## Uruchomienie (z katalogu głównego repo; torch, pandas, h5py, matplotlib, obspy)

    python -m gnome.run_poc          # etap 1: forward pass, głowica "resize", losowe wagi
    python -m gnome.train_smoke      # etap 2: krótki test uczenia signals -> akumulator (CPU, kilka minut)

## Pliki
- `loader.py` – `GnomeSignals`: `signals.csv` -> `data [B,3,2048,15]`; opcjonalnie (jeśli pliki są) `target [B,37,256,256]` = `stacks.h5`/8 i podział `split`/`kind` z `meta.csv`. Bez `stacks.h5`/`meta.csv` działa jak wcześniej.
- `model.py` – `GnomeEQNet(EQNet)`; `head="dense"` (domyślnie, `HoughDenseHead`) albo `head="resize"` (`HoughHead`, pierwszy PoC).
- `run_poc.py`, `train_smoke.py`.

## Co oryginalne, co nowe
- Oryginalne bez zmian: `EQNet` (backbone ResNet1D resnet18, `phase_picker`, `event_detector`), `log_transform`.
- Nowe: `HoughDenseHead` = Conv1d 128->16 (1x1) + BN + ReLU, max po czasie (64->8), `Linear(15*16*8 -> 32*16*16)`, 4x ConvTranspose2d (x2) + Conv2d 16->37. Gęsta warstwa wiąże wzorzec sensor/czas z pikselem kierunku.

## Kształty
`[B,3,2048,15]` -> backbone `[B,15,128,64]` -> `[B,37,256,256]` (kanał k = magnituda 5*1.25^k pT, jak `stacks.h5`).

## Zweryfikowane pliki
`stacks.h5` `x` uint8 `[630,37,256,256]` (bez atrybutów; skala /8 z README, zgodna z `base_maximum`), `meta.csv` 630 wierszy x 50 kolumn, `templates_bank.npz` `[15,37,256,256]`, `network_config_SR5.json`. `sample_id` identyczne w `signals.csv`, `meta.csv` i indeksie `stacks.h5` (0..629). To podzbiór pilota: wall 280, ladder 100, noise 150, false 100.

## Wynik testu uczenia (128 próbek train, 12 epok, 40 val; MSE w kole)
model 2,13; baseline "0" 4,59; baseline "średnia z train" 2,77. Sieć uczy się maski koła i poziomu, a zaczyna odtwarzać piki; lokalizacja maksimum jest jeszcze słaba. To nie jest model wytrenowany.

## Ograniczenia / dalsze kroki
- Mało danych (402 train), głowica ma 15,8 mln parametrów -> ryzyko przeuczenia; potrzebny pełny zbiór.
- Stacje rozróżnia tylko pozycja w wektorze (stała kolejność alfabetyczna); osie czułości i RF z `network_config_SR5.json` nie są jeszcze używane.
- Wejście to cały 600-s zapis, a cel powstaje z ostatnich 120 s; można przyciąć wejście.
- Dalej: miary dla piku (położenie maksimum, błąd kąta po `fold`), strata ważona obszarem piku, `evaluate.py` z `datasets/code`.
