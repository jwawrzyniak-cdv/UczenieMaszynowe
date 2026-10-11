# Laboratorium 3: Regresja, modele zespołowe, strojenie, interpretacja

Czas: 3 godz. lekcyjne (135 min)
Zbiór danych: Ames Housing (OpenML, `fetch_openml(name="house_prices", as_frame=True)`), 1460 domów, 79 cech, cel `SalePrice`
Środowisko: Google Colab (CPU wystarczy)

## Cel

Ćwiczenie prowadzi przez pełny cykl rozwiązywania zadania regresji: od analizy zmiennej docelowej i przetwarzania danych z uwzględnieniem znaczenia braków, przez modele liniowe z regularyzacją i modele zespołowe, po strojenie hiperparametrów, ocenę na odłożonym zbiorze testowym i interpretację wyników. Pod koniec należy umieć uzasadnić wybór metryki, wyjaśnić różnicę między baggingiem i boostingiem oraz wskazać, w których domach model myli się najbardziej i dlaczego.

## Podział czasu

| Zadanie | Temat | Czas |
|---|---|---|
| 1 | Dane i zmienna docelowa | 15 min |
| 2 | Przetwarzanie i baseline | 20 min |
| 3 | Modele liniowe i regularyzacja | 25 min |
| 4 | Modele zespołowe | 25 min |
| 5 | Strojenie hiperparametrów | 25 min |
| 6 | Interpretacja i analiza błędów | 20 min |
| 7 | Ocena końcowa i pytania kontrolne | 5 min |
| | Razem | 135 min |

## Przed zajęciami

Potrzebne są konto Google i dostęp do Colaba. Wszystkie biblioteki (scikit-learn, pandas, matplotlib, scipy) są w Colabie dostępne domyślnie, a dane pobierane są z OpenML przy pierwszym uruchomieniu, więc połączenie z internetem jest konieczne. Warto przeczytać `lab03_wprowadzenie.md`. Należy otworzyć w Colabie plik `lab03_szablon.ipynb` (menu Plik, Prześlij notebook), uruchomić komórki z sekcji Przygotowanie i sprawdzić, że wypisuje się wersja scikit-learn 1.4 lub nowsza. Numeracja nagłówków w notebooku odpowiada numeracji poleceń poniżej: nagłówek 2.3 to polecenie 3 z zadania 2.

W całym ćwiczeniu obowiązuje `RANDOM_STATE = 42`. Walidacja krzyżowa to `KFold(n_splits=5, shuffle=True, random_state=42)`, zawsze na zbiorze treningowym. Do wyboru modeli służy RMSLE z walidacji. Zbiór testowy wolno użyć dopiero w zadaniu 6, już po wyborze modelu.

## Zadanie 1: Dane i zmienna docelowa (15 min)

1. Pobierz zbiór przez `fetch_openml`. Usuń kolumnę `Id` z cech. Wypisz kształt macierzy cech (powinno być 1460 wierszy).
2. Podziel dane na zbiór treningowy i testowy w proporcji 80/20 (`train_test_split`, `random_state=42`). Od tej chwili zbiór testowy jest odłożony i nie wolno na nim niczego dopasowywać ani oglądać wyników modeli.
3. Wypisz statystyki ceny w zbiorze treningowym (`describe`). Narysuj obok siebie histogram ceny i `log1p(ceny)`, w tytułach podaj skośność (`scipy.stats.skew`). Zapisz, dlaczego uczenie na logarytmie ceny jest uzasadnione.
4. Policz korelację Spearmana cech liczbowych z `log1p(ceny)` i wypisz 10 najsilniejszych. Narysuj wykres rozrzutu `GrLivArea` względem ceny oraz `OverallQual` względem `log1p(ceny)`.
5. Wypisz domy treningowe o `GrLivArea` powyżej 4000 razem z ceną, jakością i warunkami sprzedaży (`SaleCondition`). Opisz, co je łączy i czy ich ceny są zgodne z oczekiwaniami.

## Zadanie 2: Przetwarzanie i baseline (20 min)

1. Wypisz kolumny z brakami wraz z liczbą braków w zbiorze treningowym. Dla par (`PoolQC`, `PoolArea`), (`GarageType`, `GarageArea`), (`BsmtQual`, `TotalBsmtSF`), (`FireplaceQu`, `Fireplaces`), (`MiscFeature`, `MiscVal`) sprawdź, jak często `NaN` w pierwszej kolumnie zbiega się z zerem w drugiej. Wskaż, które braki oznaczają nieobecność elementu, a które są prawdziwymi brakami informacji.
2. Szablon zawiera gotowy słownik `ORD` z kolejnością kategorii porządkowych (jakość od `Po` do `Ex`, dostępność garażu i piwnicy i inne), przepisany z opisu zbioru. Etykieta `brak` jest w nim najniższym poziomem. Sprawdź, ile kolumn jest porządkowych, a ile pozostaje nominalnych.
3. Napisz funkcję `make_preprocessor()` zwracającą `ColumnTransformer` z gałęziami: kolumny porządkowe (imputacja stałą `brak`, `OrdinalEncoder` z kolejnością z `ORD`, standaryzacja), nominalne z `NaN` oznaczającym brak elementu (imputacja `brak`, `OneHotEncoder`), nominalne z prawdziwymi brakami (imputacja najczęstszą wartością, `OneHotEncoder`), pozostałe nominalne (`OneHotEncoder` z `min_frequency=5` i `handle_unknown="infrequent_if_exist"`), liczbowe z brakami (mediana, wskaźnik braku, standaryzacja), pozostałe liczbowe (standaryzacja). Podział kolumn na grupy zrób na podstawie wyników z polecenia 1 (kolumna `MSSubClass` jest zapisana liczbą, ale jest kodem rodzaju budynku, więc traktuj ją jako nominalną). Napisz też `make_model(estimator)` zwracającą `TransformedTargetRegressor` (`log1p` i `expm1`) opakowujący `Pipeline` z preprocesorem i modelem. Dopasuj preprocesor na `X_train` i sprawdź kształt wyniku oraz to, że nie zostały żadne `NaN`.
4. Napisz funkcję `evaluate(name, model)`: `cross_validate` na zbiorze treningowym z metrykami MAE, RMSE, R2 i RMSLE (`neg_root_mean_squared_log_error`), średnie wyniki z poprawnym znakiem i średni czas dopasowania zapisane w słowniku `results`, oraz `summary()` zwracającą z niego tabelę. Szkielet jest w szablonie.
5. Oceń trzy punkty odniesienia: `DummyRegressor` (średnia i mediana), Ridge bez transformacji celu i Ridge z logarytmem celu. Policz, ile predykcji walidacyjnych (`cross_val_predict`) Ridge'a bez logarytmu jest ujemnych. Wyjaśnij, dlaczego jego RMSLE nie daje się policzyć, i porównaj MAE oraz RMSE obu wersji Ridge'a.

## Zadanie 3: Modele liniowe i regularyzacja (25 min)

1. Oceń `LinearRegression`, `Ridge`, `Lasso` i `ElasticNet` z domyślnymi parametrami (przez `make_model` i `evaluate`). Zwróć uwagę na wynik Lasso i ElasticNet i znajdź jego przyczynę.
2. Napisz funkcję liczącą krzywą walidacyjną (`validation_curve`) po parametrze `alpha` zagnieżdżonego modelu (nazwa parametru w `TransformedTargetRegressor` to `regressor__model__alpha`). Dla Ridge użyj `np.logspace(-1, 2.5, 12)` i metryki `neg_root_mean_squared_log_error`. Narysuj błąd treningowy i walidacyjny w skali logarytmicznej osi x i zapisz najlepsze `alpha`.
3. Zrób to samo dla Lasso i ElasticNet (`l1_ratio=0.5`, `alpha` z `np.logspace(-4, -1, 10)`, `max_iter=20000`). Dopisz do tabeli wyników trzy modele z najlepszym `alpha` i porównaj ich RMSLE.
4. Dopasuj preprocesor na zbiorze treningowym i policz ścieżki współczynników dla Ridge (`alpha` z `np.logspace(-1, 4, 40)`) i Lasso (`np.logspace(-4, 0, 40)`) uczonych na `log1p(y_train)`, w pętli po `alpha`. Narysuj trzy wykresy: współczynniki Ridge, współczynniki Lasso i liczbę niezerowych współczynników Lasso. Dla najlepszego `alpha` Lasso podaj, ile współczynników jest zerowych, oraz wypisz 10 największych co do modułu. Zinterpretuj współczynnik przy `GrLivArea`.
5. Wyznacz predykcje walidacyjne najlepszego Ridge'a (`cross_val_predict`) i wypisz 5 domów o największym błędzie bezwzględnym (cena, predykcja, `GrLivArea`, `SaleCondition`). Policz RMSE ze wszystkimi domami i bez dwóch największych błędów. Wyjaśnij, skąd bierze się różnica.

## Zadanie 4: Modele zespołowe (25 min)

1. Oceń `RandomForestRegressor` (200 drzew), ten sam las z `max_features=0.3`, `ExtraTreesRegressor` (200 drzew) i `HistGradientBoostingRegressor` z ustawieniami domyślnymi. Użyj `random_state=42` i `n_jobs=-1` tam, gdzie to możliwe. Porównaj RMSLE, RMSE i czas dopasowania z modelami liniowymi.
2. Narysuj krzywe walidacyjne: dla lasu losowego po `n_estimators` (10, 30, 100, 200), dla `HistGradientBoostingRegressor` po `max_iter` (25, 50, 100, 200). Pokaż błąd treningowy i walidacyjny obok siebie na wspólnej osi y. Opisz, jak zachowują się oba błędy przy zwiększaniu liczby drzew w baggingu, a jak w boostingu.
3. Oceń `HistGradientBoostingRegressor` bez `TransformedTargetRegressor` (cena surowa) i porównaj z wersją z logarytmem. Sformułuj wniosek o roli logarytmu dla modeli drzewiastych.
4. Posortuj tabelę wyników po RMSLE i narysuj wykres słupkowy poziomy wszystkich dotychczasowych modeli. Zwróć uwagę, czy ranking według RMSLE pokrywa się z rankingiem według RMSE.

## Zadanie 5: Strojenie hiperparametrów (25 min)

1. Zbuduj `HistGradientBoostingRegressor(max_iter=150)` w `make_model` i przeszukaj siatkę `GridSearchCV`: `learning_rate` w {0,05; 0,1; 0,2}, `max_leaf_nodes` w {6, 15}, `l2_regularization` w {0, 1}, czyli 12 kandydatów. Mierz czas (`time.perf_counter`) i wypisz najlepszy wynik oraz parametry. Nazwy parametrów mają prefiks `regressor__model__`.
2. Wykonaj `RandomizedSearchCV` z tym samym budżetem (`n_iter=12`) i rozkładami: `learning_rate` z `loguniform(0.02, 0.3)`, `max_leaf_nodes` z `randint(4, 32)`, `min_samples_leaf` z `randint(10, 60)`, `l2_regularization` z `loguniform(1e-3, 10)`. Pamiętaj o `random_state=42`.
3. Narysuj dla obu metod wynik walidacyjny każdego kandydata względem `learning_rate` (skala logarytmiczna osi x) i zbierz w tabeli: liczbę kandydatów, najlepszy RMSLE i czas. Odpowiedz, czy losowanie okazało się lepsze, jak szerokie jest obszar dobrych rozwiązań i czy któryś najlepszy parametr leży na brzegu przeszukiwanego zakresu.
4. Wybierz lepszy z dwóch wyników według wyniku walidacyjnego (`best_estimator_` jest już dopasowany do całego zbioru treningowego). Dopisz go do tabeli wyników (`evaluate` na `clone(...)`) i porównaj z Ridge'em po strojeniu i z `HistGradientBoostingRegressor` o ustawieniach domyślnych. Zapisz, dlaczego ten wynik jest optymistyczny i dlaczego nie wolno było do wyboru użyć zbioru testowego.

## Zadanie 6: Interpretacja i analiza błędów (20 min)

1. Model końcowy jest wybrany, więc po raz pierwszy używany jest zbiór testowy. Policz predykcje `y_pred` modelu końcowego i wypisz MAE oraz RMSLE. Porównaj RMSLE z wynikiem walidacyjnym.
2. Policz ważność permutacyjną (`permutation_importance`) pełnego modelu na zbiorze testowym, `scoring="neg_root_mean_squared_log_error"`, `n_repeats=10`. Narysuj 15 najważniejszych cech z odchyleniem standardowym. Wskaż, które cechy dominują.
3. Narysuj wykresy PDP z liniami ICE (`PartialDependenceDisplay.from_estimator`, `kind="both"`, `subsample=50`) dla `GrLivArea`, `OverallQual` i `YearBuilt`. Kolumny całkowitoliczbowe trzeba wcześniej zamienić na `float`, inaczej metoda zgłosi błąd. Opisz kształt każdej krzywej i to, co mówią linie ICE.
4. Narysuj cenę rzeczywistą względem przewidzianej (z linią y = x) oraz błąd względny `(predykcja - cena) / cena` względem ceny przewidzianej. Wypisz medianę błędu względnego i medianę jego wartości bezwzględnej. Zwróć uwagę na obszar najdroższych domów.
5. Zbierz w tabeli 8 domów z największym błędem względnym (`Neighborhood`, `SaleCondition`, `OverallQual`, `GrLivArea`, `YearBuilt`, cena, predykcja, błąd w procentach) oraz 3 domy z największym błędem w dolarach. Policz medianę bezwzględnego błędu względnego w grupach `SaleCondition`. Sformułuj hipotezę, dlaczego model myli się właśnie w tych domach, i zaznacz, czego dane nie pozwalają stwierdzić.

## Zadanie 7: Ocena końcowa (5 min)

1. Dopasuj na zbiorze treningowym `DummyRegressor` i Ridge'a z najlepszym `alpha`, weź model końcowy z zadania 5 i policz dla wszystkich trzech MAE, RMSE, R2 i RMSLE na zbiorze testowym. Dodaj kolumnę z RMSLE z walidacji krzyżowej.
2. Wyznacz 95-procentowy przedział bootstrapowy RMSLE modelu końcowego na zbiorze testowym (2000 losowań ze zwracaniem kwadratów błędów logarytmicznych). Oceń, czy różnica między modelami na teście jest większa niż niepewność oceny.
3. Napisz w komórce markdown 4 do 6 zdań: który model wybierasz, jak wypada na teście w porównaniu z walidacją, które cechy decydują o cenie i gdzie model się myli.

## Pytania kontrolne

1. Dlaczego w zadaniu regresji ceny stosuje się logarytm celu i jak zmienia to interpretację współczynników modelu liniowego?
2. Skąd bierze się różnica między MAE i RMSE tego samego modelu i kiedy jest duża?
3. Dlaczego Lasso z `alpha=1` dało wynik jak Dummy, a Ridge z tą samą wartością nie?
4. Czym różni się zerowanie współczynników w Lasso od ich zmniejszania w Ridge i kiedy które zachowanie jest korzystne?
5. Dlaczego zwiększanie liczby drzew w lesie losowym nie prowadzi do nadmiernego dopasowania, a w boostingu może?
6. Dlaczego model liniowy i model drzewiasty inaczej zachowują się wobec domu o wyjątkowo dużej powierzchni?
7. Kiedy `RandomizedSearchCV` ma przewagę nad `GridSearchCV`, a kiedy siatka jest wystarczająca?
8. Dlaczego wynik najlepszego kandydata ze strojenia jest zazwyczaj lepszy niż wynik na zbiorze testowym?
9. Co oznacza, że ważność permutacyjna cechy jest bliska zeru, i dlaczego przy cechach skorelowanych trzeba ją czytać ostrożnie?
10. Co pokazują linie ICE, czego nie widać na samym wykresie PDP?

## Do oddania

Notebook z wykonanymi wszystkimi komórkami, z zapisanymi wynikami i wykresami, wraz z odpowiedziami na pytania z poleceń (w komórkach markdown pod odpowiednimi nagłówkami) oraz komórką z wnioskami z polecenia 7.3. Przed oddaniem należy wybrać z menu Środowisko wykonawcze opcję "Uruchom ponownie i uruchom wszystko" i sprawdzić, że notebook przechodzi bez błędów. Pytania kontrolne nie wymagają pisemnej odpowiedzi, o ile prowadzący nie zdecyduje inaczej. Zadania z pliku `lab03_rozszerzone.md` są dodatkowe.
