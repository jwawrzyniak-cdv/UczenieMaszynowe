# Laboratorium 6: zadania dodatkowe

Pięć niezależnych zadań dla szybszych osób albo na pracę domową. Każde jest samodzielne, wczytuje własne dane i nie korzysta z wyników pozostałych, więc można je rozdzielić między osoby w grupie. Przewidywany czas to 30 do 60 minut na zadanie. Wszystkie dają się wykonać na samym CPU. Obowiązują te same zasady co na zajęciach: ziarno `RANDOM_STATE = 42`, podział danych przed treningiem, zbiór testowy tylko do końcowej oceny i zawsze jakiś punkt odniesienia do porównań.

Do oddania jest jeden notebook z sekcją na każde zrealizowane zadanie, z wynikami i krótkim wnioskiem.

## Zadanie 1: CNN w czystym PyTorch

Czas: około 45 minut. Colab ma preinstalowany PyTorch, więc nic nie trzeba doinstalowywać.

Weź architekturę CNN z zadania 3 (Conv2D 32, MaxPooling, Conv2D 64, MaxPooling, Flatten, Dropout 0.3, warstwa wyjściowa na 10 klas) i zapisz ją w PyTorch jako `nn.Sequential`. Dane Fashion-MNIST wczytaj przez `keras.datasets`, a do `DataLoader` przekaż tensory. Napisz ręcznie pętlę treningową: `optimizer.zero_grad()`, obliczenie straty, `loss.backward()`, `optimizer.step()`, osobno tryby `net.train()` i `net.eval()` oraz `torch.no_grad()` przy walidacji. Użyj `nn.CrossEntropyLoss`, która przyjmuje logity, więc ostatnia warstwa nie ma softmaxa. Trenuj 5 epok z Adamem (lr 1e-3, batch 256) i wypisz tabelę strat i dokładności walidacyjnej po każdej epoce.

Następnie wytrenuj tę samą sieć w Keras, na tych samych danych i z tą samą liczbą epok, i zestaw w tabeli liczbę parametrów, czas, dokładność walidacyjną i testową. Wyjaśnij, które elementy ręcznej pętli odpowiadają `compile` i `fit`, i dlaczego wyniki obu wersji nie muszą być identyczne (domyślne inicjalizacje wag w obu bibliotekach są różne).

## Zadanie 2: Klasyfikacja tekstu na IMDB, sieć kontra TF-IDF

Czas: około 45 minut.

Wczytaj `keras.datasets.imdb` z ograniczeniem do 10 000 najczęstszych słów, wyrównaj długość recenzji do 200 tokenów (`keras.utils.pad_sequences`) i wytrenuj prosty model: `Embedding(10000, 32)`, `GlobalAveragePooling1D`, `Dropout`, `Dense(1, sigmoid)`. Do walidacji użyj `validation_split`. Jeśli czas pozwala, spróbuj też `Conv1D` z `GlobalMaxPooling1D` albo warstwy `LSTM` (ta ostatnia jest wyraźnie wolniejsza na CPU, więc skróć recenzje lub dane).

Jako punkt odniesienia zbuduj klasyczny pipeline. Odtwórz tekst z indeksów słów (`imdb.get_word_index()`, pamiętaj o przesunięciu indeksów o 3), a następnie zastosuj `TfidfVectorizer` z unigramami i bigramami oraz regresję logistyczną. Porównaj dokładność testową i czas treningu. Z regresji logistycznej wypisz słowa o największych współczynnikach w obu kierunkach. Wniosek ma odpowiedzieć na pytanie, czy w tym zadaniu sieć z embeddingiem daje coś, czego nie daje liniowy model na TF-IDF, i co by się musiało zmienić (rodzaj sieci, ilość danych, wstępnie wytrenowane embeddingi), żeby przewaga się pojawiła.

## Zadanie 3: Autoenkoder do wykrywania anomalii

Czas: około 45 minut.

Na Fashion-MNIST uznaj jedną klasę za anomalię (np. spodnie, sandał lub koszulę), a pozostałe dziewięć za dane normalne. Z danych normalnych wydziel zbiór kalibracyjny. Wytrenuj autoenkoder o strukturze 784, 128, 32, 128, 784 (sigmoid na wyjściu, strata MSE) wyłącznie na obrazach normalnych. Błąd rekonstrukcji każdego obrazu testowego potraktuj jako wynik anomalii.

Oceń wykrywanie przez ROC AUC (anomalia jako klasa dodatnia). Próg ustal jako 95. percentyl błędów na zbiorze kalibracyjnym i podaj precyzję oraz czułość. Jako punkt odniesienia użyj błędu rekonstrukcji PCA z taką samą liczbą składowych (32). Powtórz eksperyment dla trzech różnych klas anomalii i zestaw wyniki w tabeli. Narysuj histogramy błędów dla danych normalnych i anomalii oraz kilka rekonstrukcji. Wniosek: od czego zależy trudność wykrycia (podobieństwo anomalii do klas normalnych) i czy nieliniowy autoenkoder wyraźnie wygrywa z PCA.

## Zadanie 4: MLP kontra gradient boosting na danych tabelarycznych

Czas: około 45 minut.

Pobierz zbiór Adult (`fetch_openml("adult", version=2)`), podziel go na treningowy i testowy (stratyfikacja) i zbuduj wspólny preprocessing w `ColumnTransformer`: standaryzacja cech liczbowych, imputacja i kodowanie one-hot cech kategorialnych (`sparse_output=False`). Porównaj trzy modele: regresję logistyczną, `HistGradientBoostingClassifier` i MLP w Keras (np. dwie warstwy gęste z Dropout i EarlyStopping na wydzielonej walidacji). Miary: ROC AUC, dokładność i czas.

Następnie powtórz porównanie na losowym podzbiorze 2000 wierszy treningowych, z tym samym zbiorem testowym. Sformułuj wniosek, kiedy w danych tabelarycznych sieć neuronowa jest potrzebna, a kiedy wystarczy prostszy model, biorąc pod uwagę jakość, czas treningu, wrażliwość na przygotowanie danych i interpretowalność.

## Zadanie 5: Harmonogram współczynnika uczenia i wielkość batcha

Czas: około 40 minut.

Na Fashion-MNIST wytrenuj MLP 256-128 (Adam, 5 epok) dla różnych wielkości batcha: 64, 256 i 1024 przy lr 1e-3, a dodatkowo 1024 z lr 4e-3 (podniesionym proporcjonalnie do batcha). Zestaw w tabeli liczbę kroków na epokę, czas treningu, stratę i dokładność walidacyjną. Zinterpretuj kompromis między liczbą aktualizacji wag a kosztem jednego kroku.

Następnie przy stałym batchu 128 porównaj harmonogramy współczynnika uczenia: stały, `ExponentialDecay`, `CosineDecay` (z `keras.optimizers.schedules`, przekazywane bezpośrednio do optymalizatora) oraz callback `ReduceLROnPlateau`. Narysuj krzywe straty walidacyjnej na jednym wykresie i opisz, jaki efekt ma zmniejszanie lr pod koniec treningu i od czego może zależeć jego wielkość.
