# Laboratorium 3: zadania dodatkowe

Pięć zadań jest niezależnych od siebie. Każde da się wykonać samodzielnie i można je przydzielić różnym osobom albo zrealizować jako pracę domową. Zadania A, B, C i E korzystają z tych samych danych i tych samych funkcji pomocniczych co laboratorium (Ames Housing, `make_preprocessor`, `make_model`, `evaluate`, `SCORING`, podział `X_train` i `X_test`, `cv`). Jeśli praca odbywa się w nowym notebooku, należy skopiować te definicje z notebooka z zadań podstawowych albo z `lab03_szablon.ipynb`. Zadanie D używa osobnego zbioru i nie wymaga niczego z laboratorium. Do każdego zadania należy dołączyć w notebooku komórkę markdown z wnioskami (3 do 5 zdań).

## Zadanie A: XGBoost i LightGBM kontra HistGradientBoosting

Biblioteki XGBoost i LightGBM są standardem w zastosowaniach przemysłowych, a ich implementacje boostingu różnią się szczegółami od `HistGradientBoostingRegressor` ze scikit-learn. Obie są domyślnie dostępne w Colabie.

1. Zbuduj trzy modele i oceń je funkcją `evaluate` w `make_model`: `HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15)`, `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.5, n_jobs=1)` i `LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=1, verbose=-1)`, wszystkie z `random_state=42`. Zestaw w tabeli MAE, RMSE, R2, RMSLE i czas dopasowania.
2. Najlepszy z trzech modeli sprawdź jednokrotnie na zbiorze testowym (MAE i RMSLE).
3. Dla LightGBM zbadaj kompromis między liczbą drzew a tempem uczenia: (0,2; 75 drzew), (0,1; 150), (0,05; 300), (0,025; 600). Opisz, jak zmieniają się RMSLE i czas.
4. Odpowiedz: czy różnice między trzema bibliotekami przekraczają to, co można przypisać szumowi walidacji? Co jeszcze, poza samą implementacją, różni konfiguracje w poleceniu 1?

## Zadanie B: SHAP, wartości globalne i lokalne

Ważność permutacyjna mówi, które cechy są ważne dla modelu jako całości. Wartości SHAP rozkładają natomiast predykcję dla pojedynczego domu na wkłady poszczególnych cech.

1. Zainstaluj bibliotekę (`!pip install -q shap`). Dopasuj preprocesor tak, by zwracał ramki pandas (`make_preprocessor().set_output(transform="pandas")`), na `X_train` i przetransformuj oba zbiory. Dopasuj `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.5)` na przetworzonym zbiorze treningowym, ze zmienną celu `np.log1p(y_train)`.
2. Utwórz `shap.TreeExplainer`, policz wartości SHAP dla zbioru testowego i sprawdź, że wartość bazowa plus suma wkładów równa się predykcji modelu.
3. Narysuj wykres `shap.plots.beeswarm` i słupkowy `shap.plots.bar` dla 15 cech. Porównaj kolejność z ważnością permutacyjną z zadania 6.2 i opisz, jaką dodatkową informację niesie beeswarm.
4. Wybierz dom o cenie najbliższej medianie zbioru testowego i narysuj dla niego `shap.plots.waterfall`. Wyjaśnij własnymi słowami, które cechy podniosły, a które obniżyły predykcję względem wartości bazowej. Pamiętaj, że wartości są w skali logarytmicznej, a cechy liczbowe są standaryzowane.
5. Narysuj `shap.plots.scatter` dla `GrLivArea`, z kolorowaniem według `OverallQual`. Oceń, czy wkład powierzchni zależy od jakości.

## Zadanie C: Przedziały predykcji z regresji kwantylowej

Predykcja punktowa nie mówi, jak bardzo model jest pewny. Regresja kwantylowa uczy model przewidywać wybrany kwantyl rozkładu ceny, więc dwa kwantyle tworzą przedział.

1. Zbuduj funkcję `quantile_model(q)` zwracającą `make_model(GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42))`. Dopasuj modele dla q = 0,05; 0,5 i 0,95 na zbiorze treningowym.
2. Na zbiorze testowym policz empiryczne pokrycie przedziału 5% do 95% (odsetek domów, których cena mieści się w przedziale), medianę szerokości przedziału i liczbę przedziałów odwróconych. Porównaj pokrycie z nominalnym 90%.
3. Narysuj domy testowe posortowane po cenie (co czwarty) jako punkty, wraz z pasmem przedziału i linią mediany. Opisz, jak szerokość przedziału zależy od ceny.
4. Skalibruj przedziały. Policz predykcje pozakładkowe obu granic na zbiorze treningowym (`cross_val_predict` z `cv`), wyznacz dla każdego domu wynik niezgodności `max(dolna - cena, cena - górna)` i weź jego kwantyl rzędu `ceil((n + 1) * 0,9) / n`. Tę wartość odejmij od dolnej i dodaj do górnej granicy na zbiorze testowym. Policz nowe pokrycie i nową szerokość.
5. Odpowiedz, ile kosztuje (w szerokości przedziału) poprawa pokrycia i dlaczego pokrycie liczone na 292 domach nie jest dokładnie równe nominalnemu.

## Zadanie D: Bike Sharing Demand i podział czasowy

Zadanie pokazuje, że losowy podział danych zawyża wynik, gdy obserwacje są uporządkowane w czasie. Używa osobnego zbioru: godzinowych wypożyczeń rowerów w Waszyngtonie w latach 2011 i 2012.

1. Pobierz `fetch_openml(data_id=42712, as_frame=True)`. Wypisz kształt danych, kilka pierwszych wierszy i potwierdź, że dane są w kolejności chronologicznej (cecha `year` równa 0 dla 2011 i 1 dla 2012).
2. Użyj modelu `HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)` i oceń średnie R2 oraz RMSE z walidacji krzyżowej przy czterech sposobach podziału: `KFold(5, shuffle=True, random_state=42)`, `KFold(5)`, `TimeSeriesSplit(5)` i `TimeSeriesSplit(5, gap=24)`. Zestaw wyniki w tabeli.
3. Dla `TimeSeriesSplit(5)` wypisz wynik w każdym podziale z osobna wraz z rozmiarem zbioru treningowego i latami, które w nim występują. Wyjaśnij, skąd niestabilność wyników w pierwszych podziałach.
4. Odłóż ostatnie 20% danych jako test czasowy, dopasuj model na pozostałych i policz R2 i RMSE. Powtórz bez cechy `year`. Opisz, co ten wynik mówi o ekstrapolacji w czasie przez modele drzewiaste.
5. Odpowiedz: dlaczego losowy podział daje wyższe R2 i dlaczego to nie jest uczciwa ocena modelu, który ma przewidywać przyszłość?

## Zadanie E: Inżynieria cech i selekcja Lassem w pipeline

Dodanie cech wyprowadzonych z istniejących (powierzchnia całkowita, wiek domu) jest klasycznym sposobem poprawy modelu, zwłaszcza liniowego. Selekcja cech musi być częścią pipeline'u, żeby nie przeciekała informacja ze zbioru walidacyjnego.

1. Napisz funkcję `add_features(df)` zwracającą kopię danych z nowymi kolumnami: `TotalSF` (suma `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`), `HouseAge` (`YrSold - YearBuilt`), `RemodAge` (`YrSold - YearRemodAdd`), `TotalBath` (pełne łazienki plus połowa łazienek połówkowych, nadziemne i w piwnicy), `PorchSF` (suma czterech kolumn powierzchni werand), `HasPool` i `HasGarage` (wskaźniki 0/1).
2. Zbuduj pipeline `FunctionTransformer(add_features)`, następnie preprocesor (nowe cechy dodaj do liczbowych bez braków; w `make_preprocessor` trzeba w tym celu dodać parametr z dodatkowymi kolumnami liczbowymi) i model, w `TransformedTargetRegressor` z logarytmem. Oceń Ridge i `HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15)` z nowymi cechami i bez nich.
3. Dodaj do pipeline'u Ridge'a krok `SelectFromModel(Lasso(alpha=0.001, max_iter=20000))` przed modelem. Oceń go w walidacji krzyżowej i na zbiorze testowym. Podaj, ile cech zostało przed selekcją i po niej oraz które z nowych cech przetrwały.
4. Odpowiedz: dlaczego poprawa po dodaniu cech jest inna dla modelu liniowego i dla boostingu i czy różnice wykraczają poza szum? Dlaczego `SelectFromModel` należy wstawić w pipeline, a nie stosować na całym `X_train` przed walidacją krzyżową?
