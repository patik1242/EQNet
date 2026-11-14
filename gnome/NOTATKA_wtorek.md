# GNOME × EQNet: wstępna ocena (notatka na wtorek)

**Zadanie:** sprawdzić, czy istniejącą sieć EQNet da się dopasować do sygnałów GNOME tak, by generowała obraz akumulatora Hougha (PoC, bez wymogu dokładności).

## Co zrobione
- **Dane:** `signals.csv` (630 próbek, 15 magnetometrów, okna 600 s) wczytane i zamienione na tensor `[B, 3, 2048, 15]`: czas × magnetometr, 3 kanały (amplituda surowa, amplituda po korekcie RF, maska piku), znaki amplitud zachowane.
- **Model:** oryginalny `EQNet` (backbone ResNet1D, oryginalne głowice bez zmian) + jedna nowa głowica. Forward pass daje obraz `[B, 1, 256, 256]`. Wyjście pochodzi z sieci EQNet, nie z niezależnego modelu.
- **Pliki GNOME** (`stacks.h5`, `meta.csv`, szablony, konfiguracja sieci) sprawdzone: `sample_id` zgodne (0–629). To podzbiór pilota.

## Wyniki
- Pierwszy obraz to tylko test architektury (losowe wagi). Osie obrazu to czas i sensor, a nie kierunek ściany.
- Krótki trening na akumulatorach z `stacks.h5` (128 próbek): MSE 2,13 vs 2,77 dla średniej mapy, bez lokalizacji piku.
- Zmiana celu na kierunek ściany (mapa ciepła, 326 próbek treningowych, 25 epok): recall@15° = 0,10 (BCE) i 0,05 (softmax), klasyczny baseline 0,89, losowy piksel 0,02. Sieć się przeuczyła lub została przy rozkładzie a priori.
- Szybkość na CPU: sieć 49 ms vs klasyczne przeliczenie 33 ms na próbkę (GPU niemierzone).

## Wniosek
**PoC działa technicznie, potencjał jest nierozstrzygnięty.** Pilot jest za mały (326 próbek treningowych przy 15,8 mln parametrów głowicy), a sieć nie zna geometrii sensorów. Wynik 0,10 vs 0,89 nie jest jeszcze wyrokiem na tę drogę.

## Dalsze kroki (jeśli kontynuujemy)
1. Pełny zbiór (~23 000 próbek; akumulatory można liczyć z pików, bez wielogigabajtowego `stacks.h5`).
2. Osie czułości i RF sensorów jako wejście (policzone z `network_config_SR5.json`, zgodne z szablonami w 100%).
3. Cel „ściana tak/nie + kierunek + magnituda” zamiast odtwarzania całego akumulatora.
4. Ocena: recall@15°, błąd kąta, fałszywe alarmy, krzywa uczenia (czy wynik rośnie z liczbą próbek), czas na GPU.

**Reguła decyzji:** warto kontynuować, jeśli na pełnym zbiorze z geometrią recall zbliży się do ok. 0,8 albo sieć wygra na słabych/wolnych ścianach. W przeciwnym razie sieć najwyżej jako uzupełnienie klasycznego programu.
