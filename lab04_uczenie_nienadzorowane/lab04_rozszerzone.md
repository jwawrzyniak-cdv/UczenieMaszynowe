# Laboratorium 4: zadania dodatkowe

Pięć zadań jest niezależnych i równoważnych. Każde zajmuje 30 do 60 minut, można je przydzielić różnym osobom albo wybrać jedno jako pracę domową. Do wykonania wystarczy ten sam notebook startowy co w laboratorium: otwórz nowy notatnik w Colabie i skopiuj z szablonu komórkę z importami oraz `RANDOM_STATE = 42`. Biblioteki spoza domyślnego zestawu Colaba są wskazane przy zadaniach, w których są potrzebne. Wynik oddaj jako notebook wykonany od początku do końca, z wykresami i krótkimi odpowiedziami w komórkach markdown.

## Zadanie A: Kwantyzacja kolorów obrazu (30 do 40 min)

Obraz kolorowy z 96 tysiącami różnych kolorów można zapisać za pomocą kilkunastu, jeśli każdy piksel zastąpić najbliższym kolorem z palety. Paletę wyznacza k-means w przestrzeni RGB.

1. Wczytaj obraz `china.jpg` przez `sklearn.datasets.load_sample_image`, zamień wartości na zakres od 0 do 1 i policz, ile ma unikalnych kolorów.
2. Dla palet o 2, 4, 8, 16, 32 i 64 kolorach dopasuj `KMeans` na losowej próbce 10 tysięcy pikseli (`random_state=42`), a potem odtwórz cały obraz. Wyświetl wyniki obok oryginału.
3. Policz błąd średniokwadratowy rekonstrukcji dla każdej wielkości palety i narysuj go względem liczby kolorów. Porównaj z paletą wylosowaną z pikseli obrazu.
4. Porównaj czas dopasowania na próbce, na pełnych danych i dla `MiniBatchKMeans`.
5. Odpowiedz: od ilu kolorów obraz wygląda dla Ciebie akceptowalnie i czy zgadza się to z krzywą błędu?

## Zadanie B: Algorytm Lloyda w NumPy (40 do 60 min)

1. Wczytaj Wholesale (`fetch_openml("wholesale-customers", as_frame=True)`), zostaw sześć kolumn wydatków (bez zdublowanej kolumny regionu i bez `Channel`), zlogarytmuj je i wystandaryzuj. Na tych danych pracuje całe zadanie.
2. Napisz funkcję `lloyd(X, centers)` w samym NumPy: przypisanie do najbliższego centroidu, przeliczenie centroidów, test zbieżności (przypisania przestają się zmieniać) i zapis inercji w każdej iteracji. Zadbaj o przypadek pustego skupienia. Narysuj inercję względem numeru iteracji.
3. Wylosuj cztery punkty danych jako centroidy startowe i porównaj wynik z `KMeans(init=centroidy_startowe, n_init=1, tol=0, algorithm="lloyd")`: liczba iteracji, inercja, największa różnica położenia centroidów i ARI między podziałami.
4. Dopisz inicjalizację `k-means++` i porównaj ją z losową: dla k równego 4 i 10 wykonaj po 50 pojedynczych uruchomień każdej (ziarna od 0 do 49) i porównaj minimum, medianę i maksimum końcowej inercji.
5. Odpowiedz: czy `k-means++` dał wyraźną poprawę na tych danych i czym tłumaczysz wynik?

## Zadanie C: HDBSCAN i klasteryzacja spektralna (40 do 50 min)

K-means i GMM zakładają zwarte, w przybliżeniu wypukłe skupienia. Sprawdź, co się dzieje, gdy ten warunek nie jest spełniony.

1. Wygeneruj `make_moons` (`noise=0.07`), `make_circles` (`factor=0.5`, `noise=0.05`) oraz trzy skupienia o różnej gęstości przez `make_blobs` (różne `cluster_std`).
2. Dla każdego zbioru uruchom `KMeans`, `AgglomerativeClustering` (ward), `GaussianMixture`, `DBSCAN`, `sklearn.cluster.HDBSCAN` i `SpectralClustering` (`affinity="nearest_neighbors"`). Podaj ARI względem prawdziwych etykiet i narysuj podziały w siatce wykresów.
3. Zbadaj wrażliwość: DBSCAN dla kilku wartości `eps`, HDBSCAN dla kilku wartości `min_cluster_size`.
4. Odpowiedz: która metoda wygrywa na którym zbiorze i co to mówi o założeniach algorytmów? Wskaż, który parametr trudniej dobrać bez znajomości etykiet.

## Zadanie D: Reguły asocjacyjne (40 do 50 min)

Wymaga biblioteki `mlxtend`. Zainstaluj aktualną wersję: `!pip install -q -U mlxtend`.

Dane zakupów nie są częścią laboratorium, więc wygeneruj je sam. Zbiór ma być jawnie syntetyczny: opisz to w notebooku, bo reguły znalezione w danych losowych nie mówią nic o prawdziwych klientach.

1. Z ustalonym ziarnem (`np.random.default_rng(42)`) wygeneruj około 1500 koszyków po 12 produktów (zmienne zero-jedynkowe). Wpleć w generator kilka celowych zależności, na przykład że kupujący chipsy częściej biorą piwo.
2. Znajdź zbiory częste algorytmem `apriori` (`min_support` około 0,05) i wyznacz reguły przez `association_rules` (miara `lift`, próg około 1,2).
3. Wypisz 10 reguł o najwyższym `lift` i sprawdź, czy odtworzyły zależności z generatora. Zinterpretuj jedną regułę o wysokim `support`, ale niskim `lift`.
4. Zbadaj wrażliwość: ile reguł zostaje dla `min_support` równego 0,02, 0,05, 0,1 i 0,2.
5. Odpowiedz: czym różnią się `support`, `confidence` i `lift` i dlaczego wysoka `confidence` sama o niczym nie świadczy?

## Zadanie E: One-Class SVM kontra IsolationForest (30 do 40 min)

1. Wczytaj Wholesale (`fetch_openml("wholesale-customers", as_frame=True)`), zostaw sześć kolumn wydatków (bez zdublowanej kolumny regionu i bez `Channel`), zlogarytmuj je i wystandaryzuj.
2. Dopasuj `IsolationForest` (`contamination=0.05`) oraz `OneClassSVM` (`kernel="rbf"`, `nu=0.05`) dla `gamma` równego `"scale"`, 0,01, 0,05, 0,2 i 0,5. Dla każdego ustawienia podaj liczbę oznaczonych klientów, liczbę wspólnych z IsolationForest i korelację rang wyników.
3. Zbuduj `Pipeline` z `OneClassSVM`, dopasuj go na części treningowej (70 procent) i porównaj odsetek oznaczonych obserwacji w części treningowej i testowej dla `nu` równego 0,01, 0,05 i 0,1.
4. Odpowiedz: co kontroluje `nu`, jak `gamma` zmienia granicę decyzji i dlaczego ta metoda jest czulsza na dobór hiperparametrów niż IsolationForest?
