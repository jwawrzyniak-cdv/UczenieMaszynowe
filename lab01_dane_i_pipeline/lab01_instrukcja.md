# Laboratorium 1: Proces uczenia maszynowego, przygotowanie danych, Pipeline

Czas: 2 godz. lekcyjne (90 min)
Zbiór danych: Adult (OpenML), `fetch_openml("adult", version=2, as_frame=True)`
Środowisko: Google Colab (CPU wystarczy)

## Cel

Przejdź cały proces uczenia maszynowego na jednym zbiorze: od sformułowania problemu, przez audyt i podział danych, po pipeline z przetwarzaniem i modelem oraz jego zapis. Zobacz na liczbach, czym jest wyciek informacji i które decyzje przetwarzania naprawdę zmieniają wynik, a które mieszczą się w szumie. Zbiór dotyczy tego, czy roczny dochód osoby przekracza 50 tys. USD.

## Podział czasu

| Zadanie | Temat | Czas |
|---|---|---|
| 1 | Sformułowanie problemu | 10 min |
| 2 | Wczytanie i audyt danych | 15 min |
| 3 | Podział danych i wyciek informacji | 15 min |
| 4 | ColumnTransformer i Pipeline | 25 min |
| 5 | Wpływ decyzji przetwarzania | 15 min |
| 6 | Zapis pipeline'u i predykcja | 10 min |
| | Razem | 90 min |

## Przed zajęciami

Przeczytaj `lab01_wprowadzenie.md`. Na zajęciach potrzebne jest konto Google i dostęp do Colaba, a runtime może być domyślny (CPU). Wszystkie biblioteki (scikit-learn, pandas, numpy, matplotlib, joblib) są w Colabie zainstalowane, więc niczego nie doinstalowujesz. Dane pobierasz programowo z OpenML, bez wgrywania plików.

Otwórz `lab01_szablon.ipynb` w Colabie. Nagłówki w notebooku odpowiadają numeracji poniżej. Przed oddaniem uruchom całość od początku (Środowisko wykonawcze, Uruchom wszystko) i sprawdź, czy przechodzi bez błędów.

## Zadanie 1: Sformułowanie problemu (10 min)

Zanim zaczniesz pisać kod, ustal, co dokładnie robisz i po czym poznasz sukces. Wpisz w notebooku krótką notatkę (komórka tekstowa, najlepiej tabela), w której dla zbioru Adult określisz:

1. Cel biznesowy: kto i po co chciałby przewidywać dochód. Wymyśl jedno realistyczne zastosowanie i zapisz je w dwóch zdaniach.
2. Zmienną docelową i jej kodowanie oraz rodzaj problemu.
3. Jednostkę obserwacji: co oznacza jeden wiersz.
4. Metrykę sukcesu (główną i uzupełniające) wraz z uzasadnieniem. Od razu zastanów się, czy samo accuracy wystarczy.
5. Koszty błędów: co kosztuje w Twoim zastosowaniu fałszywy alarm, a co pominięcie osoby o wysokim dochodzie. Nie podawaj liczb, których nie znasz. Opisz, jak by je oszacowano.
6. Baseline: jaki wynik uzyska rozwiązanie bez żadnej wiedzy o danych. Liczby uzupełnisz po zadaniu 2.
7. Ryzyka związane ze zbiorem (wskazówka: rok powstania danych i cechy wrażliwe).

## Zadanie 2: Wczytanie i audyt danych (15 min)

1. Wczytaj zbiór (`fetch_openml("adult", version=2, as_frame=True, return_X_y=True)`). Zakoduj cel jako 0/1, przy czym 1 oznacza dochód powyżej 50 tys. Zapisz listy nazw kolumn liczbowych i kategorycznych (`num_cols`, `cat_cols`), bo użyjesz ich w kolejnych zadaniach. Wypisz rozmiar, typy kolumn i kilka pierwszych wierszy.
2. Zbadaj braki. Ile ich jest w każdej kolumnie, czy braki w różnych kolumnach współwystępują i czy udział klasy `>50K` jest taki sam w rekordach z brakami i bez braków?
3. Pokaż rozkład zmiennej docelowej i oblicz accuracy klasyfikatora, który zawsze wskazuje klasę większościową. Narysuj dwa wykresy: rozkład wieku w obu klasach oraz udział klasy `>50K` w zależności od `education-num`.
4. Policz duplikaty (same cechy oraz cechy razem z celem). Sprawdź, czy identyczne wiersze cech mają czasem różne etykiety.
5. Zbadaj cechy podejrzane: `fnlwgt` (liczba różnych wartości, korelacja z celem), para `education` i `education-num` (czy to ta sama informacja?), `capital-gain` i `capital-loss` (udział zer, wartości skrajne) oraz `sex` i `race` (udziały grup i udział klasy `>50K` w grupach płci).

Po tym zadaniu uzupełnij baseline w notatce z zadania 1. Dla każdej wykrytej cechy podejrzanej zapisz jedno zdanie: zostawiasz, usuwasz czy obserwujesz dalej i dlaczego.

## Zadanie 3: Podział danych i wyciek informacji (15 min)

1. Podziel dane na treningowe i testowe (80/20) ze stratyfikacją względem celu i `random_state=RANDOM_STATE`. Sprawdź rozmiary oraz udział klasy `>50K` w całości, w treningu i w teście.
2. Pokaż wyciek przy skalowaniu. Dopasuj `StandardScaler` na cechach liczbowych (a) całego zbioru i (b) samego zbioru treningowego. Zestaw w tabeli średnie i odchylenia z obu wariantów i policz różnicę procentową. Przeskaluj zbiór testowy obiema wersjami skalera i podaj największą różnicę wartości w każdej kolumnie. Następnie powtórz porównanie na losowych podpróbkach o rozmiarach 100, 200, 1000, 5000 i 20000 wierszy (po 200 powtórzeń, podział 80/20, mediana średniej bezwzględnej różnicy po skalowaniu).
3. Pokaż wyciek przez cechę. Do kopii zbioru dodaj kolumnę `tax_class`, równą etykiecie, ale odwróconą w losowych 3% rekordów (generator `np.random.default_rng(RANDOM_STATE)`). Sprawdź korelacje cech liczbowych z celem na zbiorze treningowym. Wytrenuj `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` na cechach liczbowych bez `tax_class` i z nią, a następnie porównaj accuracy i ROC AUC na zbiorze testowym.

W tym zadaniu zbiór testowy służy wyłącznie do demonstracji. W zadaniach 4 i 5 wybierasz wszystko na zbiorze treningowym.

## Zadanie 4: ColumnTransformer i Pipeline (25 min)

1. Zbuduj preprocessor (`ColumnTransformer`). Część liczbowa: `SimpleImputer(strategy="median")` i `StandardScaler`. Część kategoryczna: `SimpleImputer(strategy="most_frequent")` i `OneHotEncoder(handle_unknown="ignore", min_frequency=50)`. Obie części złóż w `Pipeline`.
2. Zbuduj trzy pipeline'y z tym samym preprocessorem: `LogisticRegression(max_iter=1000)`, `DummyClassifier(strategy="most_frequent")` i `DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)`.
3. Oceń je w pięciokrotnej stratyfikowanej walidacji krzyżowej (`StratifiedKFold` z `shuffle=True`) na zbiorze treningowym, używając `cross_validate` i metryk: accuracy, balanced accuracy, F1 i ROC AUC. Zbierz średnie i odchylenia standardowe w jednej tabeli.
4. Dopasuj pipeline z regresją logistyczną na zbiorze treningowym. Odczytaj nazwy cech po przetworzeniu (`get_feature_names_out`), podaj ich liczbę i wypisz po sześć cech o najmniejszych i największych współczynnikach. Następnie sprawdź wpływ `min_frequency` (None, 10, 50, 500) na liczbę cech i na ROC AUC w walidacji krzyżowej.

## Zadanie 5: Wpływ decyzji przetwarzania (15 min)

1. Napisz funkcję `make_pipe(model, scale=True, cat_fill="most_frequent", drop=())`, która buduje pełny pipeline z możliwością wyłączenia skalowania, zmiany imputacji kategorii na stałą `"brak"` i usunięcia wskazanych kolumn. Przygotuj słownik wariantów: regresja logistyczna (wzorcowa, bez skalowania, braki jako `"brak"`, bez `fnlwgt`, bez `education`, bez obu, `class_weight="balanced"`) oraz drzewo decyzyjne z `max_depth=8` (ze skalowaniem i bez).
2. Oceń każdy wariant w tej samej walidacji krzyżowej na zbiorze treningowym. Policz przy okazji ostrzeżenia `ConvergenceWarning` (`warnings.catch_warnings(record=True)`). Zbierz w tabeli średnie metryk, odchylenie ROC AUC, czas dopasowania i liczbę ostrzeżeń. Narysuj wykres słupkowy ROC AUC.
3. Dla każdego wariantu policz różnicę ROC AUC względem wzorca w każdym foldzie (te same podziały, więc foldy są sparowane). Podaj średnią różnicę, minimum, maksimum i liczbę foldów, w których wariant wygrywa.

Wnioski wyciągaj, porównując różnice z odchyleniem standardowym między foldami.

## Zadanie 6: Zapis pipeline'u i predykcja (10 min)

1. Wybierz wariant na podstawie tabeli z zadania 5 i uzasadnij wybór w jednym akapicie. Dopasuj go na całym zbiorze treningowym i oceń jeden raz na zbiorze testowym (accuracy, balanced accuracy, F1, ROC AUC) obok klasyfikatora `DummyClassifier`. Pokaż macierz pomyłek i `classification_report`.
2. Zapisz pipeline razem z numerem wersji scikit-learn za pomocą `joblib.dump`, odczytaj go i sprawdź, czy predykcje na zbiorze testowym są identyczne.
3. Zbuduj jednowierszowy `DataFrame` z nową osobą (nazwy kolumn jak w `X`) i podaj prawdopodobieństwo klasy `>50K`. Sprawdź jeszcze rekord z brakiem w `workclass` i nieznaną wartością `native-country` oraz rekord, w którym brakuje całej kolumny.

## Pytania kontrolne

Odpowiedz w notebooku, po jednym lub dwóch zdaniach.

1. Dlaczego accuracy równe 0,85 nie jest w tym zadaniu równie imponujące, jak mogłoby się wydawać?
2. Co odróżnia wyciek informacji od zwykłego nadmiernego dopasowania? Podaj przykład z tego laboratorium.
3. Dlaczego regresja logistyczna wymaga skalowania, a drzewo decyzyjne nie?
4. Co by się stało, gdyby w danych produkcyjnych pojawiły się braki w kolumnach liczbowych? Czy pipeline jest na to gotowy?
5. Dlaczego zapisujemy cały pipeline, a nie sam model, i co jeszcze warto zapisać razem z plikiem?
6. Którą decyzję przetwarzania uznajesz za najważniejszą w tym zadaniu i na jakiej podstawie?

## Do oddania

Notebook `.ipynb` z wykonanymi komórkami (wszystkie wyniki widoczne), zawierający notatkę z zadania 1, kod i wyniki zadań 2 do 6 oraz odpowiedzi na pytania kontrolne. Notebook ma przechodzić "Uruchom wszystko" bez błędów. Udostępnij go przez link w Colabie lub wyeksportuj do pliku (Plik, Pobierz, `.ipynb`).
