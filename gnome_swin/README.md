# GNOME × EQNet: adaptacja do akumulatora Hougha (PoC)

Sygnały GNOME (piki z `datasets/signals.csv`) -> zaadaptowany `EQNet` -> obraz 256×256 (akumulator Hougha).
Katalog jest samodzielny: oryginalne pliki repozytorium (`eqnet/`) i dane (`datasets/`) nie są zmieniane.
Wyniki i wnioski: `NOTATKA_wtorek.md`.

Wszystkie polecenia uruchamiaj z katalogu głównego repo (`EQNet`).

## Pliki
| plik | co robi |
|---|---|
| `loader.py` | `signals.csv` -> tensor `[B, 3, nt, 15]` (czas × magnetometr). Okno **120 s** (starsze piki odrzucane), 3 kanały: amplituda surowa, po korekcie RF, maska piku; znaki zachowane; sensory w kolejności alfabetycznej |
| `geometry.py` | oś czułości (w układzie Ziemi) i RF każdego sensora z `network_config_SR5.json` -> `[15, 4]` |
| `model.py` | `GnomeEQNetSwin` (backbone SwinTransformerV2), `GnomeEQNetResNet` (backbone ResNet1D z `EQNet`), `HoughHead`, `add_geometry`, `station_locations` |
| `run_poc.py` | test techniczny bez treningu (losowe wagi): kształty, obraz, zależność między stacjami |
| `data_ideal.py` | dane testu na idealnych sygnałach: wejście z pików + cel z `datasets/stacks.h5`, błąd kąta maksimum |
| `test_ideal.py` | trening i ocena (przeuczenie na kilku próbkach albo train -> val/test) |
| `logs/` | logi eksperymentów (`logs/earlier/` – wcześniejsze) |

## Model
- **Oryginalne z repo:** `EQNet` (klasa bazowa, oryginalne głowice `phase_picker` i `event_detector` dostępne przez `original_heads=True`), `ResNet1D`, `SwinTransformerV2`, `log_transform`.
- **Dodane:** `HoughHead` = Conv2d(128->64, 3×3) + GroupNorm + ReLU + Conv2d(64->1) + skalowanie do 256×256 (GroupNorm zamiast BatchNorm: `train()` i `eval()` dają ten sam wynik).
- **Zmienione względem repo:** pierwsza warstwa backbone (miała na sztywno 3 kanały wejścia) przy włączonej geometrii ma 3+4 kanały.
- **Swin:** okno uwagi = [`window_t`, 16] (16 = 15 magnetometrów + 1 pusta stacja, bo `PatchMerging` dopełnia oś stacji do parzystej; przy oknie mniejszym od osi stacji wynik zależał od kolejności kolumn). Pozycje stacji `(lon, lat)/180` w biasie pozycji.
- **ResNet1D** liczy każdą stację osobno; stacje łączy dopiero głowica. Swin łączy stacje w uwadze.
- Kształty: `[B, 3(+4), nt, 15]` -> backbone `[B, 15, 128, nt/32]` -> `HoughHead` -> `[B, 1, 256, 256]`.

## Uruchamianie
Test techniczny (bez treningu, losowe wagi):

    python -m gnome_swin.run_poc

Test na idealnych danych (bg = `ideal`, 66 ścian po odrzuceniu "don't care"):

    # przeuczenie 8 próbek, cel = jeden kanał akumulatora, z geometrią
    python -m gnome_swin.test_ideal --backbone resnet --mode overfit --n 8 --target channel --geometry --epochs 100
    python -m gnome_swin.test_ideal --backbone swin --mode overfit --n 8 --target channel --geometry --nt 512 --epochs 100

    # uogólnianie: uczenie na train (45 ścian), ocena na val i test
    python -m gnome_swin.test_ideal --backbone resnet --mode train --target channel --geometry --epochs 80

Opcje: `--nt` (biny czasu na 120 s, domyślnie 2048), `--win_t` (okno Swin w czasie, domyślnie 7), `--sd` (stochastic depth Swin, domyślnie 0), `--norm group|batch`, `--lr`, `--clip`, `--threads`.
Wypisywane miary: MSE (eval i train-mode), korelacja z celem, mediana błędu kąta maksimum, odsetek próbek poniżej 15°, zależność cech jednej stacji od pozostałych (przed i po treningu) oraz punkt odniesienia "średnia mapa".

## Wymagania
Python z `torch`, `pandas`, `numpy`, `h5py`, `matplotlib`, `obspy` (import `eqnet` ciągnie moduły danych). Na Windowsie przy polskich znakach: `PYTHONIOENCODING=utf-8`.

## Ograniczenia
- To PoC. Obraz z losowych wag nie jest akumulatorem Hougha.
- Dane to pilot (630 próbek, 70 idealnych ścian); wyniki na 8 próbkach to przeuczenie, a nie uogólnianie.
- Cel testu to akumulator referencyjny z `hough4gnome` (spłaszczony do jednego kanału), nie prawdziwy kierunek; kierunek służy tylko do oceny.
- Krok gradientu przy `--bs 8` i 8 próbkach to cała epoka, więc "epoki" = kroki.
- Swin: koszt rośnie z kwadratem rozmiaru okna (okno 14 ~60 s na krok na CPU); bias pozycji dostaje tylko (lon, lat), a osie czułości trafiają do sieci przez kanały wejścia.
- Brak szumu, brak pełnego zbioru (~23 000 próbek).
