# toy-real-transformer-pytorch

Minimalny transformer w PyTorch — z prawdziwym treningiem przez gradient descent.

Companion repo do [toy-fake-transformer-numpy](https://github.com/protechnologia/toy-fake-transformer-numpy) — ta sama architektura, ale tym razem wagi są naprawdę uczone.

## Po co to?

Żeby zobaczyć różnicę między "ustawieniem wag ręcznie" a "nauczeniem wag przez trening". Architektura identyczna jak w wersji numpy — zmienił się tylko sposób wyznaczania wag.

## Jak uruchomić

```bash
pip install torch
python toy_transformer.py
```

Oczekiwany output:

```
Trening:

  epoch    0  loss=1.7865  acc=33%
  epoch  500  loss=0.0000  acc=100%
  ...
  epoch 2999  loss=0.0000  acc=100%

Po treningu:

  ✓  [Ala] → ma  (oczekiwano: ma)
  ✓  [Ala ma] → kota  (oczekiwano: kota)
  ✓  [Ala ma kota] → a  (oczekiwano: a)
  ✓  [Ala ma kota a] → kot  (oczekiwano: kot)
  ✓  [Ala ma kota a kot] → ma  (oczekiwano: ma)
  ✓  [Ala ma kota a kot ma] → Alę  (oczekiwano: Alę)
```

## Co się zmieniło względem wersji numpy

| | toy-fake-transformer-numpy | toy-real-transformer-pytorch |
|---|---|---|
| Biblioteka | numpy | PyTorch |
| Embeddingi `E` | jednostkowe (one-hot) | losowe, uczone |
| `Wq`, `Wk`, `Wv` | jednostkowe (atrapa) | losowe, uczone |
| `W_lm` | lstsq (analitycznie) | Adam (gradient descent) |
| Attention | atrapa | prawdziwe |
| Backprop | brak | `loss.backward()` |

## Architektura

```
token + pozycja → attention (causal) → residual → LM head → next token
```

| Komponent | Wartość |
|---|---|
| Słownik | 6 tokenów |
| Wymiar embeddingu | 8 |
| Warstwy attention | 1 |
| Optymalizator | Adam, lr=0.05 |
| Epoki | 3000 |

## Jak działa trening

Teacher forcing — model dostaje całą sekwencję wejściową naraz i jednocześnie uczy się przewidywać następny token na każdej pozycji:

```
wejście:  [Ala, ma, kota, a,   kot, ma ]  (tokens[0..5])
cele:     [ma,  kota, a,  kot, ma,  Alę]  (tokens[1..6])
```

Loss to średnia cross-entropy po wszystkich 6 pozycjach. `loss.backward()` liczy gradienty przez całą sieć — przez `W_lm`, residual, attention, `Wq`/`Wk`/`Wv`, aż do embeddingów `E`. Adam aktualizuje wszystkie parametry po każdym kroku.

## Dlaczego "real"?

W wersji numpy `Wq=Wk=Wv=I` sprawiało że attention było atrapą — każdy token patrzył tylko na siebie. Tu `Wq`, `Wk`, `Wv` są losowo zainicjalizowane i naprawdę uczone — model sam odkrywa jak ustawić Q, K, V żeby skutecznie przewidywać sekwencję.

## Dlaczego "toy"?

Nadal ta sama sekwencja 7 tokenów i słownik 6 słów. Prawdziwy transformer, ale na zabawkowych danych.

## Kodowanie pozycyjne a embeddingi

`E` (embeddingi tokenów) to "co jestem" — uczony. `pos_enc` to "gdzie jestem" — stałe sinusoidy. Suma daje "co jestem i gdzie jestem":

```
E[ma]       = [-1.14,  1.39,  1.14, ...,  0.00,  0.00]   ← uczony, różny dla każdego tokenu
pos_enc(1)  = [ 0.00,  0.00,  0.00, ...,  0.50,  0.87]   ← stały, różny dla każdej pozycji
──────────────────────────────────────────────────────
suma        = [-1.14,  1.39,  1.14, ...,  0.50,  0.87]   ← unikalny dla ("ma", pozycja 1)
```

```
E[ma]       = [-1.14,  1.39,  1.14, ...,  0.00,  0.00]   ← ten sam embedding
pos_enc(5)  = [ 0.00,  0.00,  0.00, ...,  0.50, -0.87]   ← inna pozycja, inny cos
──────────────────────────────────────────────────────
suma        = [-1.14,  1.39,  1.14, ...,  0.50, -0.87]   ← unikalny dla ("ma", pozycja 5)
```

Token `ma` pojawia się dwa razy w sekwencji — `E[ma]` jest zawsze identyczne, ale `pos_enc` różni się znakiem `cos`. Model widzi różne wektory i może poprowadzić je w różne strony — pierwszy "ma" → "kota", drugi "ma" → "Alę".
