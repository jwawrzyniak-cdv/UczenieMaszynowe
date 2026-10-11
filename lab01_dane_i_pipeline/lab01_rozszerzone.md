# Laboratorium 1: zadania dodatkowe

Zadania są alternatywami, nie kolejnymi etapami jednego projektu. Każde da się zrobić samodzielnie, w osobnym notebooku (lub w osobnej sekcji notebooka z laboratorium). Prowadzący może przydzielić różnym osobom różne zadania. Przy każdym z nich zacznij od wczytania zbioru Adult, zakodowania celu jako 0/1 i stratyfikowanego podziału 80/20 z `random_state=42`. Wszystkie oceny wykonuj w pięciokrotnej stratyfikowanej walidacji krzyżowej na zbiorze treningowym, a zbioru testowego użyj najwyżej raz, na końcu.

Każde zadanie kończy się pytaniem, na które odpowiedz w komórce tekstowej, odwołując się do liczb z tabeli.

## A. Imputacja na sztucznie wprowadzonych brakach

Cechy liczbowe Adult nie mają braków, więc znasz wartości prawdziwe i możesz zmierzyć błąd imputacji.

1. Weź losową próbkę 8000 wierszy i sześć kolumn liczbowych. Usuń losowo 10% komórek (maska z `np.random.default_rng(42)`).
2. Wystandaryzuj dane (skaler dopasowany na danych z lukami, `StandardScaler` pomija NaN), zaimputuj je czterema imputerami: `SimpleImputer` (średnia i mediana), `KNNImputer(n_neighbors=5)` i `IterativeImputer(max_iter=10, random_state=42)` (wymaga `from sklearn.experimental import enable_iterative_imputer`).
3. Dla każdej kolumny policz RMSE tylko na usuniętych komórkach, w jednostkach odchylenia standardowego kolumny. Dołóż czas działania każdego imputera.
4. Sprawdź macierz korelacji cech. Narysuj wykres słupkowy RMSE w podziale na kolumny.
5. Zmierz wpływ na klasyfikację: w 10000 wierszach wprowadź braki w cechach liczbowych i porównaj ROC AUC `make_pipeline(imputer, StandardScaler(), LogisticRegression())` dla czterech imputerów oraz dla danych bez braków.

Pytanie: czy bardziej złożony imputer wygrywa ze średnią i dlaczego tak lub nie? Powiąż odpowiedź z macierzą korelacji.

## B. Własny transformer

1. Napisz klasę `CapitalFeatures(BaseEstimator, TransformerMixin)` z parametrami `use_log` i `add_flags`. Transformer przyjmuje kolumny `capital-gain` i `capital-loss` i zwraca logarytm `log1p` obu oraz flagi, czy wartość jest dodatnia. Zapamiętaj nazwy kolumn w `fit`, zaimplementuj `get_feature_names_out` i użyj `check_is_fitted` w `transform`.
2. Sprawdź kompatybilność: `clone`, `get_params`, działanie w `ColumnTransformer` z `set_output(transform="pandas")` oraz błąd przy użyciu przed `fit`.
3. Włącz transformer do pipeline'u (pozostałe cechy liczbowe i kategoryczne przetwórz jak w laboratorium) i porównaj ROC AUC oraz F1 dla wariantów: wzorzec, log z flagami, tylko log, tylko flagi. Zrób to dla regresji logistycznej i dla drzewa decyzyjnego (`max_depth=8`).

Pytanie: który element transformera daje zysk i dlaczego drzewo na tym nie korzysta?

## C. Kodowanie cechy o wysokiej liczności

1. Zmieniaj sposób kodowania wyłącznie dla `native-country` (41 poziomów, ok. 90% to United-States), a pozostałe kolumny kategoryczne koduj stale `OneHotEncoder`. Porównaj: OneHot ze wszystkimi poziomami, OneHot z `min_frequency=100`, `OrdinalEncoder` (z `handle_unknown="use_encoded_value"`) i `TargetEncoder`. Oceń regresję logistyczną i drzewo decyzyjne.
2. Wykonaj eksperyment z wyciekiem: utwórz kolumnę `kod` z 2000 losowych poziomów (niezwiązaną z celem). Porównaj ROC AUC trzech podejść, używając tylko tej kolumny: (a) średnia celu dla poziomu policzona raz na całym zbiorze treningowym, a potem walidacja krzyżowa, (b) `TargetEncoder` wewnątrz pipeline'u, (c) `OneHotEncoder` wewnątrz pipeline'u. Poprawna metoda powinna dać wynik bliski 0,5.

Uwaga techniczna: od scikit-learn 1.9 parametr `random_state` w `TargetEncoder` jest wycofywany, a losowość ustawia się przez przekazanie generatora walidacji (`cv=KFold(5, shuffle=True, random_state=42)`). W starszych wersjach `cv` przyjmuje tylko liczbę, a `random_state` działa normalnie.

Pytanie: dlaczego podejście (a) daje wynik wyraźnie lepszy od losowego, skoro kolumna jest szumem?

## D. Wartości odstające w pipeline'ie

1. Napisz transformer `QuantileClipper(upper)`, który w `fit` zapamiętuje górny kwantyl każdej kolumny, a w `transform` przycina wartości do tej granicy. Zaimplementuj `get_feature_names_out`.
2. Wypisz kwantyle 50, 90, 95, 99% i maksimum cech liczbowych na zbiorze treningowym. Policz, jaki odsetek wartości przycinają kwantyle 0,99 i 0,95 w każdej kolumnie.
3. Porównaj w walidacji krzyżowej cztery warianty części liczbowej: `StandardScaler`, przycięcie 0,99 i `StandardScaler`, przycięcie 0,95 i `StandardScaler`, `RobustScaler`. Zrób to dla regresji logistycznej i drzewa decyzyjnego. Zapisz też czas dopasowania i liczbę ostrzeżeń `ConvergenceWarning`.

Pytanie: dlaczego przycinanie kwantylem 0,95 szkodzi w tym zbiorze, a w innych często pomaga? Co mówią kwantyle `capital-gain` i `capital-loss`?

## E. Karta zbioru danych

Przygotuj dokument (jedna do dwóch stron, w notebooku jako markdown) opisujący zbiór Adult z punktu widzenia osoby, która ma zdecydować, czy wolno go użyć w danym zastosowaniu. Liczby do karty zbierz z danych.

1. Pochodzenie i zawartość: skąd pochodzą dane, rok, liczba rekordów i cech, rozkład celu, braki, duplikaty, cechy specjalne (`fnlwgt`, `capital-gain`). Sprawdź i podaj źródła (strona zbioru w UCI, praca Ding, Hardt, Miller, Schmidt, Retiring Adult, NeurIPS 2021, https://arxiv.org/abs/2108.04884).
2. Struktura grup: liczebność i udział klasy `>50K` w grupach `sex` i `race`, odsetek rekordów z brakami według rasy.
3. Model referencyjny (regresja logistyczna w pipeline'ie, bez `fnlwgt`) i raport per grupa na zbiorze testowym: liczba rekordów, udział przewidzianych `>50K`, czułość (TPR), odsetek fałszywych alarmów (FPR), precyzja. Zaznacz grupy, dla których liczebność w teście jest zbyt mała na wnioski.
4. Sprawdź, czy usunięcie kolumny `sex` zmienia raport per grupa płci, i zmierz, z jakim ROC AUC da się odgadnąć płeć z pozostałych cech. Zajrzyj do rozkładu płci w `relationship`.
5. Napisz sekcje karty: pochodzenie, zawartość, ograniczenia, ryzyka etyczne, zalecane i niezalecane zastosowania.

Pytanie: czy usunięcie cechy wrażliwej z danych wystarcza, by model nie różnicował grup? Uzasadnij liczbami z punktu 4.
