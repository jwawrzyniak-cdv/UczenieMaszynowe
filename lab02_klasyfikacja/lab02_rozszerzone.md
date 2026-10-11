# Laboratorium 2: zadania dodatkowe

Pięć zadań alternatywnych dla szybszych osób lub do pracy domowej. Zadania są od siebie niezależne: każde zajmuje 30 do 60 minut i można wykonać dowolne z nich, w dowolnej kolejności. Zadania A, B, D i E używają zbioru German Credit, zadanie C zbioru digits. Rozwiązania zapisujcie w osobnym notebooku, jedna sekcja na zadanie.

## Wspólne przygotowanie

Zadania na German Credit zaczynają się od tego samego kodu. Część importów (`imblearn`, `scipy`) jest potrzebna tylko w wybranych zadaniach.

```python
%pip install -q imbalanced-learn

credit = fetch_openml("credit-g", version=1, as_frame=True)
X = credit.data
y = (credit.target == "bad").astype(int)   # 1 = zły klient
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

num_cols = X_train.select_dtypes("number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

def make_pipe(clf):
    return Pipeline([("prep", preprocess), ("clf", clf)])

def mean_cost(y_true, y_pred):             # koszt na klienta, FN kosztuje 5, FP kosztuje 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (fp + 5 * fn) / len(y_true)
```

Zbiór testowy `X_test`, `y_test` w tych zadaniach służy co najwyżej do kontrolnego porównania na końcu. Wybory i oceny robicie na zbiorze treningowym.

## Zadanie A: Oversampling i wyciek danych (ok. 45 min)

Gdy klasa pozytywna jest rzadka, sięga się czasem po sztuczne zwiększenie jej liczności, na przykład metodą SMOTE, która tworzy nowe przykłady jako interpolacje między sąsiadami z klasy rzadkiej. Łatwo to zrobić źle.

1. Oceńcie las losowy (`n_estimators=200`) i kNN (`n_neighbors=5`) przez `cross_validate` z `RepeatedStratifiedKFold(5, 3)`, metrykami recall, f1 i roc_auc, bez próbkowania.
2. Powtórzcie ocenę, wstawiając SMOTE do `imblearn.pipeline.Pipeline` po preprocessingu: `[("prep", preprocess), ("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", model)]`. Pipeline z imblearn stosuje próbkowanie wyłącznie w fazie uczenia, osobno w każdym podziale.
3. Teraz błędnie: zakodujcie cały zbiór treningowy (`preprocess.fit_transform`), zastosujcie `SMOTE.fit_resample` na całości i dopiero wtedy wykonajcie walidację krzyżową na powiększonym zbiorze.
4. Dopasujcie modele z punktów 1 do 3 na zbiorze treningowym (bez próbkowania, z SMOTE w pipeline i na danych z punktu 3) i policzcie recall, f1 i pole pod ROC na zbiorze testowym.
5. Zestawcie w jednej tabeli oszacowania z walidacji i wyniki testowe dla trzech wariantów i dwóch modeli.

Wyjaśnijcie, dlaczego oszacowanie z punktu 3 jest zawyżone i które z oszacowań zgadza się z wynikiem testowym. Zastanówcie się też, czy SMOTE poprawił ranking klientów (pole pod ROC), czy tylko przesunął punkt pracy klasyfikatora, i co jeszcze dałoby podobny efekt.

## Zadanie B: Kalibracja prawdopodobieństw (ok. 45 min)

Model może dobrze porządkować klientów i jednocześnie podawać prawdopodobieństwa, które nie odpowiadają rzeczywistym częstościom. Dla progu z macierzy kosztów (1/6) ma to znaczenie.

1. Przygotujcie warianty: las losowy bez kalibracji, las z `CalibratedClassifierCV(..., method="sigmoid", cv=5)`, las z `method="isotonic"`, regresja logistyczna oraz SVC z kalibracją sigmoidalną. Każdy model owińcie najpierw w `make_pipe`, a dopiero potem w `CalibratedClassifierCV`.
2. Policzcie predykcje z walidacji krzyżowej (`cross_val_predict(..., method="predict_proba")`) dla każdego wariantu.
3. Zbudujcie tabelę: wynik Briera (`brier_score_loss`), logarytmiczna strata (`log_loss`), pole pod ROC, koszt na klienta przy progu 0,5 i przy progu 1/6.
4. Narysujcie krzywe kalibracji (`calibration_curve`, `n_bins=8`, `strategy="quantile"`) wszystkich wariantów na jednym wykresie, z przekątną odniesienia.

Który wariant jest najlepiej skalibrowany według wyniku Briera i według wykresu? Co dzieje się z kalibracją izotoniczną przy tak małym zbiorze? Czy kalibracja zmienia koszt biznesowy bardziej niż zmiana progu z 0,5 na 1/6?

## Zadanie C: Klasyfikacja wieloklasowa (ok. 40 min)

Zadanie na zbiorze cyfr `sklearn.datasets.load_digits` (1797 obrazów 8 na 8 pikseli, 10 klas).

1. Porównajcie regresję logistyczną (`max_iter=2000`, ze skalowaniem), kNN (`n_neighbors=5`, ze skalowaniem) i las losowy przez `cross_validate` z `StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)`. Metryki: accuracy, f1_micro, f1_macro, f1_weighted, precision_macro, recall_macro.
2. Dla regresji logistycznej policzcie `cross_val_predict`, wypiszcie `classification_report` i narysujcie macierz pomyłek 10 na 10. Które cyfry są najczęściej mylone?
3. Zróbcie wersję niezbalansowaną: zostawcie 20 procent przykładów cyfry 8 i 10 procent cyfry 9 (losowanie z `np.random.default_rng(RANDOM_STATE)`), pozostałe klasy bez zmian. Powtórzcie punkty 1 i 2 na tych danych.

Czym różnią się średnie mikro, makro i ważone i kiedy mikro-F1 jest równe accuracy? Co ukrywa accuracy po zmniejszeniu dwóch klas i które metryki to ujawniają? Który model ucierpiał najbardziej?

## Zadanie D: Formalne porównanie dwóch modeli (ok. 50 min)

Różnica 0,01 pola pod ROC między dwoma modelami może być prawdziwa albo przypadkowa. Zadanie pokazuje dwa sposoby rozstrzygnięcia.

1. Policzcie pole pod ROC regresji logistycznej, lasu losowego, SVC i drzewa (`max_depth=5`) na tych samych 50 podziałach (`RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)`). Przekazanie tego samego obiektu walidacji do `cross_validate` gwarantuje identyczne podziały dla wszystkich modeli.
2. Dla par (regresja logistyczna, las), (las, SVC), (regresja logistyczna, drzewo) policzcie zwykły sparowany test t (`scipy.stats.ttest_rel`) i test t ze skorygowaną wariancją (Nadeau i Bengio), w którym t = średnia różnic podzielona przez pierwiastek z `(1/k + n_test/n_train) * wariancja różnic`, gdzie k to liczba podziałów (50), a n_test i n_train to liczności części walidacyjnej i treningowej w pojedynczym podziale. Wartość p to `2 * stats.t.sf(abs(t), df=k - 1)`.
3. Wykonajcie test McNemara dla regresji logistycznej i lasu na predykcjach z walidacji (próg 0,5): policzcie, w ilu przypadkach tylko pierwszy model jest poprawny (b) i tylko drugi (c), a wartość p to dokładny test dwumianowy `stats.binomtest(min(b, c), b + c, 0.5).pvalue`.
4. Zapiszcie wniosek dla każdej pary jednym zdaniem.

Dlaczego zwykły test t jest zbyt pewny siebie dla wyników z walidacji krzyżowej? Jaki jest praktyczny wniosek z braku istotnej różnicy między modelami?

## Zadanie E: Karta scoringowa, Naive Bayes i boosting (ok. 60 min)

Banki często budują kartę scoringową: regresję logistyczną na cechach pogrupowanych w przedziały, której współczynniki przelicza się na punkty. Takie punkty da się wyjaśnić klientowi i audytorowi.

1. Zbudujcie preprocessing dla karty: cechy `duration`, `credit_amount` i `age` dzielicie na 4 przedziały kwantylowe (`KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile")`), pozostałe cechy kodujecie metodą one-hot. Dopasujcie regresję logistyczną (na przykład `C=0.3`).
2. Przeliczcie współczynniki na punkty: `factor = 20 / np.log(2)` (20 punktów na podwojenie ilorazu szans), punkty kategorii `= -factor * współczynnik` (więcej punktów oznacza mniejsze ryzyko), punkty bazowe `= 600 - factor * wyraz_wolny`. Zaokrąglijcie punkty do całości.
3. Pokażcie 8 kategorii o największym ujemnym i 8 o największym dodatnim wkładzie. Wypiszcie granice przedziałów (`bin_edges_`) i zinterpretujcie wyniki.
4. Sprawdźcie, że wynik policzony z punktów (baza plus suma punktów aktywnych kategorii) zgadza się z `600 - factor * decision_function` z dokładnością do zaokrągleń. Wyznaczcie punktową granicę przyznania kredytu odpowiadającą progowi 1/6 w prawdopodobieństwie: `600 - factor * np.log(1/5)`.
5. Porównajcie przez `RepeatedStratifiedKFold(5, 3)` (pole pod ROC i średnia precyzja) oraz na predykcjach z walidacji (koszt na klienta przy progu 1/6 i 0,5): kartę, zwykłą regresję logistyczną, `GaussianNB` (z `make_pipe`) i `HistGradientBoostingClassifier(categorical_features="from_dtype")` na surowych cechach.

Czy przedziałowanie cech pogorszyło jakość względem zwykłej regresji logistycznej? Które współczynniki karty są sprzeczne z intuicją i dlaczego nie wolno ich czytać przyczynowo? Co dzieje się z kosztem Naive Bayes przy progu 1/6 i dlaczego?

## Do oddania

Notebook z rozwiązaniami wybranych zadań (zapisane wyniki, wykresy i odpowiedzi na pytania).
