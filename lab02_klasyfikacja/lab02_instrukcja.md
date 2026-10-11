# Laboratorium 2: Klasyfikacja i rzetelna ocena modeli

Czas: 3 godz. lekcyjne (135 min)
Zbiór danych: German Credit (OpenML, `fetch_openml("credit-g", version=1, as_frame=True)`)
Środowisko: Google Colab (CPU wystarczy)

## Cel

Laboratorium pokazuje, jak porównać kilka klasyfikatorów na zbiorze, w którym błędy mają różny koszt, i jak ocenić je tak, żeby wynikowi można było zaufać. Na przykładzie kredytów widać, dlaczego accuracy bywa mylące, jak czytać macierz pomyłek oraz krzywe ROC i precision-recall i jak dobrać próg decyzyjny do macierzy kosztów. Na końcu wybrany model jest jednorazowo oceniany na odłożonym zbiorze testowym.

## Podział czasu

| Zadanie | Temat | Czas |
|---|---|---|
| 1 | Dane i macierz kosztów | 15 min |
| 2 | Pięć klasyfikatorów i baseline | 30 min |
| 3 | Metryki klasyfikacji | 25 min |
| 4 | Próg decyzyjny i koszt biznesowy | 25 min |
| 5 | Nadmierne i niedostateczne dopasowanie | 20 min |
| 6 | Ocena końcowa | 15 min |
| 7 | Pytania kontrolne i podsumowanie | 5 min |
| | Razem | 135 min |

## Przed zajęciami

Przeczytajcie `lab02_wprowadzenie.md`. Otwórzcie `lab02_szablon.ipynb` w Google Colab (runtime CPU). Biblioteki (scikit-learn, pandas, numpy, matplotlib) są w Colabie domyślnie, więc nic nie trzeba instalować. Dane pobierają się z OpenML przy pierwszym wywołaniu `fetch_openml`, co wymaga internetu i zajmuje zwykle kilka sekund. W szablonie nagłówki odpowiadają numeracji poniżej, a każde zadanie rozwiązujecie w jego sekcji.

Zbiór testowy wydzielacie w zadaniu 1 i sięgacie po niego dopiero w zadaniu 6. Wszystko pomiędzy (wybór modelu, progu, hiperparametrów) opiera się wyłącznie na zbiorze treningowym.

## Zadanie 1: Dane i macierz kosztów (15 min)

1. Wczytajcie zbiór: `credit = fetch_openml("credit-g", version=1, as_frame=True)`. Cechy to `credit.data`, cel to `credit.target` (wartości `good` i `bad`). Zakodujcie cel jako liczby, przyjmując za klasę pozytywną złego klienta: `y = (credit.target == "bad").astype(int)`.
2. Sprawdźcie rozmiar danych, typy cech, braki oraz proporcje klas. Ilu klientów jest złych, w procentach?
3. Opis zbioru (`credit.DESCR`) zawiera macierz kosztów. Przeczytajcie ją i zapiszcie koszty jako stałe: przyznanie kredytu złemu klientowi to 5, odmowa dobremu 1. Napiszcie funkcję `mean_cost(y_true, y_pred)` zwracającą średni koszt na klienta, na przykład na podstawie `confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()`, która daje kolejno TN, FP, FN, TP. Zbudujcie z niej scorer: `make_scorer(mean_cost, greater_is_better=False)`.
4. Policzcie koszt na klienta dla dwóch strategii trywialnych: kredyt dla wszystkich (przewidujemy same 0) i odmowa dla wszystkich (same 1).
5. Wydzielcie zbiór testowy: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)`. Sprawdźcie, że proporcje klas w obu częściach są takie same.
6. Przygotujcie preprocessing w `Pipeline`: cechy liczbowe skalowane, kategorialne kodowane metodą one-hot. Szkielet z laboratorium 1:

```python
num_cols = X_train.select_dtypes("number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

def make_pipe(clf):
    return Pipeline([("prep", preprocess), ("clf", clf)])
```

W notebooku powinny się znaleźć proporcje klas, dwie liczby kosztu dla strategii trywialnych oraz rozmiary zbiorów treningowego i testowego. Zastanówcie się, dlaczego "odmowa dla wszystkich" jest dla banku tańsza niż "kredyt dla wszystkich" i jakiego wyniku należy oczekiwać od sensownego modelu.

## Zadanie 2: Pięć klasyfikatorów i baseline (30 min)

1. Zdefiniujcie słownik `models` z sześcioma klasyfikatorami: `DummyClassifier(strategy="most_frequent")`, `LogisticRegression(max_iter=1000)`, `KNeighborsClassifier(n_neighbors=15)`, `DecisionTreeClassifier(max_depth=5)`, `RandomForestClassifier(n_estimators=200)` i SVC. Tam, gdzie to ma znaczenie, ustawcie `random_state=RANDOM_STATE`. SVC samo nie zwraca prawdopodobieństw, a będą potrzebne w zadaniach 3 i 4, więc użyjcie wariantu z kalibracją: `CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False)`.
2. Każdy model opakujcie w `make_pipe` i oceńcie przez `cross_validate` z `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)` na zbiorze treningowym. Metryki: accuracy, balanced_accuracy, precision, recall, f1, roc_auc, average_precision oraz własny koszt. Dla precision podajcie `make_scorer(precision_score, zero_division=0)`, bo Dummy nie przewiduje klasy pozytywnej.
3. Zbudujcie tabelę, w której wiersze to modele, a komórki mają postać "średnia ± odchylenie standardowe". Koszt pokażcie jako wartość dodatnią (scorer zwraca ją ze znakiem minus).
4. Narysujcie wykresy pudełkowe dla pola pod krzywą ROC i dla kosztu, po jednym pudełku na model (każde z 15 wyników).

Do oceny: tabela i wykres. Które różnice między modelami przekraczają odchylenie standardowe, a które nie? Dlaczego żaden model nie schodzi przy progu 0,5 poniżej kosztu strategii "odmowa dla wszystkich", choć accuracy jest wyższe niż u Dummy?

## Zadanie 3: Metryki klasyfikacji (25 min)

1. Dla każdego modelu policzcie predykcje prawdopodobieństwa klasy 1 na zbiorze treningowym: `cross_val_predict(make_pipe(clf), X_train, y_train, cv=skf, method="predict_proba")[:, 1]`, gdzie `skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)`. Zapiszcie je w słowniku `oof`. Predykcje pochodzą z modeli, które nie widziały danego przykładu, więc zbioru testowego nie ruszacie.
2. Dla regresji logistycznej i lasu losowego narysujcie macierze pomyłek przy progu 0,5 (`ConfusionMatrixDisplay.from_predictions`) i wypiszcie `classification_report` dla regresji logistycznej. Odczytajcie TP, FP, FN, TN.
3. Narysujcie krzywe ROC wszystkich modeli oprócz Dummy (w legendzie pole pod krzywą) i, obok, krzywe precision-recall (w legendzie średnia precyzja), z poziomą linią na poziomie udziału klasy pozytywnej.

Ilu złych klientów regresja logistyczna wykrywa przy progu 0,5, a ilu przeocza? Czym różni się obraz modeli na krzywej ROC od obrazu na krzywej precision-recall? Co znaczy "dobry model", gdy klasy są niezbalansowane, a błędy mają różne koszty? Odpowiedzi zapiszcie pod wykresami.

## Zadanie 4: Próg decyzyjny i koszt biznesowy (25 min)

1. Dla siatki progów `np.linspace(0.02, 0.98, 97)` policzcie koszt na klienta na predykcjach `oof` każdego modelu (poza Dummy): klasa 1, gdy `oof[name] >= próg`.
2. Zbudujcie tabelę: koszt przy progu 0,5, najlepszy próg, koszt przy najlepszym progu. Obok wypiszcie próg teoretyczny `COST_FP / (COST_FP + COST_FN)`.
3. Narysujcie koszt w funkcji progu dla wszystkich modeli. Zaznaczcie minima, próg 0,5 i poziom kosztu strategii "odmowa dla wszystkich".
4. Dodajcie dwa modele z wagami klas: `LogisticRegression(class_weight={0: 1, 1: 5})` i `RandomForestClassifier(class_weight="balanced")`. Policzcie ich predykcje `oof` i zestawcie w tabeli z ich wersjami bez wag: koszt przy 0,5, najlepszy próg, koszt przy nim.
5. Dla regresji logistycznej wypiszcie macierz pomyłek przy progu 0,5 i przy progu minimalizującym koszt.

Jak zmieniają się FN i FP po obniżeniu progu i dlaczego koszt mimo to maleje? Czy najlepsze progi zgadzają się z teoretycznym i dlaczego nie muszą? Dlaczego minimalny koszt odczytany z tych samych predykcji, na których wybierano próg, jest nieco zbyt optymistyczny? Czym wagi klas różnią się od zmiany progu?

## Zadanie 5: Nadmierne i niedostateczne dopasowanie (20 min)

1. Narysujcie krzywe uczenia (`learning_curve`, `train_sizes=np.linspace(0.1, 1.0, 8)`, `cv=skf`, `scoring="roc_auc"`) dla trzech modeli: regresji logistycznej, drzewa bez ograniczenia głębokości i lasu losowego. Na każdym panelu pokażcie wynik treningowy i walidacyjny (średnia, a dla walidacji także pasmo odchylenia).
2. Narysujcie krzywą walidacyjną (`validation_curve`, `scoring="roc_auc"`) dla głębokości drzewa od 1 do 15 i dla parametru C regresji logistycznej na siatce `np.logspace(-4, 2, 7)`. Nazwy parametrów w pipeline mają prefiks kroku: `clf__max_depth`, `clf__C`.
3. Odczytajcie optymalną głębokość i optymalne C.

Który model ma wysoką wariancję, a który wysokie obciążenie i po czym to widać? Czy dokładanie danych poprawiłoby któryś z nich? Dlaczego las losowy, mimo wyniku treningowego równego 1,0, generalizuje dużo lepiej niż pojedyncze drzewo?

## Zadanie 6: Ocena końcowa (15 min)

1. Na podstawie samych wyników z walidacji (zadania 2 do 5) wybierzcie jeden model i jeden próg. Zapiszcie je w zmiennych i uzasadnijcie wybór w komórce tekstowej, zanim sięgniecie po zbiór testowy.
2. Dopasujcie wybrany pipeline na całym zbiorze treningowym, policzcie `predict_proba` na zbiorze testowym i zastosujcie wybrany próg. Robicie to raz.
3. Podajcie koszt na klienta i koszt łączny, `classification_report` oraz macierz pomyłek.
4. Oszacujcie niepewność kosztu przez bootstrap: 2000 losowań ze zwracaniem spośród 200 klientów testowych i 95-procentowy przedział z percentyli 2,5 i 97,5.
5. Porównajcie koszt testowy z kosztem strategii trywialnych i z kosztem tego samego modelu przy progu 0,5. To ostatnie porównanie jest tylko opisowe: nie wolno na jego podstawie zmieniać wyboru.

Czy koszt testowy zgadza się z oczekiwaniem z walidacji, a jeśli nie, to czy rozbieżność mieści się w niepewności? Dlaczego przy 200 klientach nie można zbyt pewnie twierdzić, że model bije strategię trywialną?

## Zadanie 7: Pytania kontrolne i podsumowanie (5 min)

Odpowiedzcie w notebooku na pytania z następnej sekcji, po jednym do trzech zdań na pytanie.

## Pytania kontrolne

1. Dlaczego klasyfikator, który zawsze przyznaje kredyt, ma accuracy 0,70 i co z tego wynika dla oceny modeli na tym zbiorze?
2. Dlaczego próg dobieramy na predykcjach z walidacji krzyżowej, a nie na zbiorze treningowym ani testowym?
3. Kiedy krzywa precision-recall jest bardziej informatywna niż ROC?
4. Skąd wynika teoretyczny próg 1/6 i dlaczego empiryczne progi różnią się od niego w zależności od modelu?
5. Jak na krzywej uczenia odróżnić niedostateczne dopasowanie od nadmiernego?
6. Dlaczego wynik na zbiorze testowym może być gorszy od wyniku z walidacji mimo poprawnej procedury i czego nie wolno wtedy zrobić?

## Do oddania

Notebook `lab02_szablon.ipynb` po uzupełnieniu: z zapisanymi wynikami komórek, wykresami i krótkimi wnioskami po każdym zadaniu. Przed oddaniem uruchomcie go od początku do końca (Środowisko wykonawcze, Uruchom wszystko) i sprawdźcie, że nie ma błędów.
