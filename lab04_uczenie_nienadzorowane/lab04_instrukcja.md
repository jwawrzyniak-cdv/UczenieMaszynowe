# Laboratorium 4: Uczenie nienadzorowane

Czas: 3 godz. lekcyjne (135 min)
Zbiór danych: Wholesale customers (OpenML, `fetch_openml("wholesale-customers", as_frame=True)`) oraz digits (`sklearn.datasets.load_digits`)
Środowisko: Google Colab (CPU wystarczy)

## Cel

Hurtownia ma 440 klientów i chce przekazać działowi handlowemu opis kilku typów odbiorców. Nikt nie podał etykiet, więc segmenty trzeba wyznaczyć z samych wydatków. Zadania prowadzą od przygotowania danych przez k-means i inne algorytmy skupień, po PCA, wizualizację t-SNE i wykrywanie nietypowych klientów. Na końcu masz umieć uzasadnić wybór liczby segmentów, opisać je w jednostkach zrozumiałych dla handlowca i powiedzieć, czego analiza nie rozstrzyga.

## Podział czasu

| Zadanie | Temat | Czas |
|---|---|---|
| 1 | Dane i przygotowanie do skupień | 15 min |
| 2 | K-means | 25 min |
| 3 | Inne algorytmy skupień | 30 min |
| 4 | PCA | 20 min |
| 5 | Wizualizacja wysokowymiarowa | 20 min |
| 6 | Wykrywanie anomalii | 20 min |
| 7 | Podsumowanie | 5 min |
| | Razem | 135 min |

## Przed zajęciami

Przeczytaj `lab04_wprowadzenie.md`. Otwórz `lab04_szablon.ipynb` w Colabie (Plik, Wczytaj notatnik) i ustaw dowolny runtime CPU. Pierwsza komórka instaluje `umap-learn`, który nie jest domyślnie dostępny. Potrzebny jest dostęp do internetu, bo oba zbiory są pobierane programowo. Wszystkie losowe elementy mają używać `RANDOM_STATE = 42`. Przy każdym zadaniu w szablonie znajdziesz nagłówki i komórki z komentarzami, co uzupełnić.

## Zadanie 1: Dane i przygotowanie do skupień (15 min)

### 1.1 Wczytanie i kontrola danych

1. Pobierz zbiór z OpenML i obejrzyj kolumny, ich typy oraz kilka pierwszych wierszy. Porównaj nazwy kolumn z opisem w `DESCR`.
2. Jedna z kolumn powtarza informację z innej. Znajdź ją, usuń i nadaj kolumnom wydatków czytelne nazwy zgodne z opisem zbioru.
3. Oddziel `Channel` i `Region` od cech wydatków. Zmienne te nie wchodzą do skupień, `Channel` posłuży później do porównań. Sprawdź typ tych kolumn po wczytaniu i w razie potrzeby zamień je na liczby całkowite.
4. Sprawdź braki i duplikaty oraz wypisz statystyki opisowe.

### 1.2 Skośność i transformacja logarytmiczna

1. Policz skośność każdej kolumny wydatków, a potem po przekształceniu `np.log1p`.
2. Narysuj histogramy w dwóch wierszach (surowe, po logarytmie).
3. Napisz, dla której kategorii zmiana jest największa i dlaczego rozkład wydatków jest skośny.

### 1.3 Skalowanie i odległość euklidesowa

1. Zlogarytmuj dane i wystandaryzuj je (`Xs`). Pamiętaj, że w tym zadaniu nie ma podziału na treningowy i testowy, bo nie oceniamy przewidywań na nowych danych.
2. Dla klientów o indeksach 0 i 1 policz odległość euklidesową przed i po przygotowaniu. Dla każdej cechy podaj, jaki procent kwadratu odległości ona wnosi.
3. Policz też średni udział każdej cechy w kwadracie odległości po wszystkich parach klientów. Skomentuj różnicę między danymi surowymi a przygotowanymi.

### 1.4 Wpływ przygotowania na k-means

1. Uruchom `KMeans` z k = 4 na trzech wariantach danych: surowych, tylko wystandaryzowanych oraz zlogarytmowanych i wystandaryzowanych.
2. Dla każdego wariantu podaj rozmiary skupień oraz ARI względem `Channel`.
3. Wyjaśnij, dlaczego rozmiary skupień w wariantach różnią się tak bardzo i który wariant uznajesz za właściwy do dalszej pracy.

## Zadanie 2: K-means (25 min)

### 2.1 Metoda łokcia i współczynnik sylwetki

1. Dla k od 2 do 10 dopasuj `KMeans` (`n_init=10`) i zapisz `inertia_` oraz `silhouette_score`.
2. Narysuj oba wykresy obok siebie i wypisz tabelę wartości.

### 2.2 Stabilność względem ziarna losowego

1. Dla k od 2 do 6 uruchom `KMeans` dziesięć razy z różnymi `random_state`, osobno dla `n_init=1` i `n_init=10`.
2. Porównaj każdy podział z pierwszym za pomocą ARI i podaj minimum ARI oraz rozrzut inercji dla każdego k.
3. Skomentuj, jak `n_init` i liczba skupień wpływają na powtarzalność wyniku.

### 2.3 Wybór k i profil skupień

1. Wybierz k i uzasadnij wybór, odwołując się do wyników z 2.1 i 2.2 oraz do tego, co da się sensownie opisać handlowcom. Dopasuj końcowy model i zapisz etykiety jako `seg`.
2. Policz średnie i mediany wydatków w każdym skupieniu w jednostkach oryginalnych (nie w danych zlogarytmowanych).
3. Narysuj wykres słupkowy średnich i mapę cieplną stosunku średnich w skupieniu do średniej ogólnej.

### 2.4 Nazwy segmentów i porównanie z kanałem

1. Zbuduj tabelę krzyżową skupień i kanału sprzedaży (Horeca, Retail).
2. W osobnej komórce markdown nadaj segmentom nazwy i opisz każdy dwoma zdaniami w języku zrozumiałym dla handlowca.

### 2.5 Przydział nowego klienta

1. Zbuduj `Pipeline`: `log1p`, `StandardScaler`, `KMeans` i dopasuj go do surowych wydatków. Sprawdź, że daje ten sam podział co wcześniej (ARI równe 1).
2. Wymyśl dwóch klientów (wydatki w sześciu kategoriach), przydziel ich do segmentów i sprawdź, czy wynik zgadza się z Twoją intuicją.

## Zadanie 3: Inne algorytmy skupień (30 min)

### 3.1 Klasteryzacja hierarchiczna

1. Dla czterech wartości `linkage` (`ward`, `complete`, `average`, `single`) wykonaj `AgglomerativeClustering` z k = 4. Porównaj rozmiary skupień, sylwetkę i ARI względem `Channel`.
2. Narysuj obcięty dendrogram dla metody Warda (`scipy.cluster.hierarchy.linkage` i `dendrogram`) i zaznacz poziom, na którym cięcie daje cztery skupienia.
3. Wyjaśnij, dlaczego sylwetka nie wystarcza do wyboru metody łączenia. Zwróć uwagę na rozmiary skupień.

### 3.2 DBSCAN

1. Narysuj wykres k-odległości dla k równego `min_samples = 10` i odczytaj zakres sensownych wartości `eps`.
2. Sprawdź wrażliwość: dla kilku wartości `eps` podaj liczbę skupień i liczbę punktów szumu.
3. Wybierz `eps`, dopasuj DBSCAN i zapisz etykiety jako `labels_db`. Porównaj mediany wydatków punktów szumu z resztą klientów.

### 3.3 Mieszanina rozkładów normalnych

1. Dla kowariancji `full` oraz `diag` policz BIC dla od 1 do 8 składowych (`n_init=5`) i narysuj krzywe.
2. Wybierz liczbę składowych, dopasuj model i zapisz etykiety jako `labels_gmm`.
3. Za pomocą `predict_proba` policz klientów, których najwyższe prawdopodobieństwo przynależności jest mniejsze niż 0,8. Zastanów się, co to znaczy dla działu handlowego.

### 3.4 Porównanie metod

1. Zbierz w jednej tabeli dla k-means, Warda, DBSCAN i GMM: liczbę skupień, liczbę punktów szumu, sylwetkę (bez szumu) i ARI względem `Channel`.
2. Policz macierz ARI między samymi metodami.
3. Napisz, co oznacza niskie lub wysokie ARI względem `Channel`. Podział na kanały jest zewnętrznym punktem odniesienia, nie prawdą ostateczną: segmenty mogą słusznie dzielić klientów inaczej niż kanał.

## Zadanie 4: PCA (20 min)

### 4.1 Wyjaśniona wariancja

1. Dopasuj PCA do `Xs` i narysuj wyjaśnioną wariancję (słupki) oraz jej sumę skumulowaną (krzywa).
2. Podaj, ile składowych potrzeba na około 70 i 80 procent wariancji.
3. Dla porównania uruchom PCA na surowych wydatkach bez skalowania i sprawdź udział oraz ładunki pierwszej składowej.

### 4.2 Ładunki składowych

1. Zbuduj tabelę ładunków (`pca.components_`) trzech pierwszych składowych.
2. Zinterpretuj każdą z nich słowami: które kategorie rosną razem, które się przeciwstawiają.

### 4.3 Skupienia w przestrzeni dwóch składowych

1. Zrzutuj `Xs` na dwie pierwsze składowe. Narysuj dwa wykresy obok siebie: kolor według skupień k-means i kolor według `Channel`. Dorysuj strzałki ładunków cech.
2. Uruchom k-means na trzech pierwszych składowych i porównaj z podziałem na wszystkich cechach za pomocą ARI.
3. Oceń, czy dwuwymiarowy rzut wystarcza, by wiarygodnie zobaczyć segmenty.

## Zadanie 5: Wizualizacja wysokowymiarowa (20 min)

### 5.1 Digits i PCA

1. Wczytaj `load_digits`, pokaż kilka obrazków i policz, ile składowych PCA potrzeba na 80, 90 i 95 procent wariancji.
2. Zapisz rzut na dwie pierwsze składowe jako `E_pca`.

### 5.2 t-SNE i perplexity

1. Uruchom `TSNE` z `init="pca"` dla `perplexity` równego 5, 30 i 100. Narysuj trzy wykresy obok siebie, kolor według cyfry, i zmierz czas każdego przebiegu. Osadzenia zapisz w słowniku `embeddings`.
2. Narysuj rzut PCA z kolorami cyfr. Porównaj go z t-SNE ilościowo: dla każdego dwuwymiarowego obrazu policz dokładność klasyfikatora 5-NN w walidacji krzyżowej.
3. Sprawdź, czy odległości między skupieniami na wykresie t-SNE coś znaczą. Policz odległości między centroidami klas w przestrzeni 64-wymiarowej, na rzucie PCA i na dwóch przebiegach t-SNE (`init="random"`, różne ziarna), a następnie porównaj je korelacją.

### 5.3 UMAP (opcjonalnie)

Jeśli zostaje czas, zbuduj osadzenie UMAP, narysuj je i policz tę samą dokładność 5-NN. Zanotuj czas działania.

## Zadanie 6: Wykrywanie anomalii (20 min)

### 6.1 IsolationForest

1. Dopasuj `IsolationForest` do `Xs` (`contamination=0.05`). Wynik anomalii zapisz tak, by większa wartość oznaczała klienta bardziej nietypowego.
2. Wypisz 10 najbardziej nietypowych klientów z wydatkami w jednostkach oryginalnych.

### 6.2 LocalOutlierFactor

1. Dopasuj `LocalOutlierFactor` (`n_neighbors=20`, `contamination=0.05`) przez `fit_predict`. Wynik anomalii weź z `negative_outlier_factor_` ze zmienionym znakiem.
2. Wypisz 10 najbardziej nietypowych klientów.

### 6.3 Zgodność metod

1. Policz, ilu klientów oznaczyła każda metoda i ilu obie. Podaj współczynnik Jaccarda i korelację rang wyników (Spearman).
2. Narysuj wykres rozrzutu wyników obu metod.

### 6.4 Interpretacja

1. Dla kilku klientów oznaczonych przez obie metody pokaż percentyle ich wydatków w każdej kategorii. Napisz, co czyni ich nietypowymi z punktu widzenia handlowca.
2. Uruchom `IsolationForest` na surowych wydatkach i sprawdź, ilu z oznaczonych klientów pokrywa się z wersją na danych zlogarytmowanych i wystandaryzowanych.

## Zadanie 7: Podsumowanie (5 min)

Zapisz w komórce markdown krótkie podsumowanie: ostateczna segmentacja (liczba i nazwy segmentów), rekomendacja dla działu handlowego, jedna lub dwie grupy klientów wymagające osobnego potraktowania (anomalie) oraz ograniczenia analizy.

## Pytania kontrolne

1. Dlaczego k-means na surowych wydatkach daje inny podział niż po transformacji logarytmicznej i standaryzacji? Która kategoria dominowała w odległości?
2. Czemu wartość współczynnika sylwetki nie wystarcza do wyboru k? Co jeszcze zbadałaś lub zbadałeś?
3. Co mówi niskie ARI między skupieniami a kanałem sprzedaży i czy to dowodzi, że skupienia są złe?
4. Dlaczego DBSCAN z jednym ustawieniem `eps` może dać jedno skupienie, a z innym prawie same punkty szumu?
5. Czym PCA różni się od t-SNE i do czego nie wolno używać odległości na wykresie t-SNE?
6. Dlaczego IsolationForest i LOF nie wskazują dokładnie tych samych klientów i jak to wykorzystać w praktyce?

## Do oddania

Notebook `lab04_szablon.ipynb` z uzupełnionymi komórkami, zapisany po wykonaniu od początku do końca ("Uruchom wszystko"), z widocznymi wynikami i wykresami. Notebook ma zawierać odpowiedzi na pytania z poleceń (w komórkach markdown pod odpowiednimi zadaniami), nazwy i opisy segmentów oraz podsumowanie z zadania 7. Nazwij plik `lab04_nazwisko.ipynb`.
