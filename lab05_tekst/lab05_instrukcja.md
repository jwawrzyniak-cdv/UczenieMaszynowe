# Laboratorium 5: Przetwarzanie tekstu, klasyfikacja tekstu, modelowanie tematów

Czas: 3 godz. lekcyjne (135 min)
Zbiór danych: PolEmo 2.0 (polskie opinie, CSV z Hugging Face) i 20 Newsgroups (scikit-learn)
Środowisko: Google Colab, CPU wystarczy

## Cel

Zajęcia prowadzą przez cały tor pracy z tekstem: od eksploracji korpusu i zamiany dokumentów na macierz cech, przez klasyfikację sentymentu polskich opinii z rzetelną oceną modeli, po modelowanie tematów, wyszukiwanie dokumentów i porównanie TF-IDF z embeddingami zdań. Główny wątek dotyczy tekstu polskiego, więc osobno ocenia się wpływ fleksji na reprezentację. Wyniki trzeba interpretować: patrzeć na cechy, które model uznał za ważne, i na przykłady, w których się myli.

## Podział czasu

| Zadanie | Temat | Czas |
|---|---|---|
| 1 | Wczytanie i eksploracja korpusu | 15 min |
| 2 | Reprezentacja tekstu | 20 min |
| 3 | Klasyfikacja sentymentu | 30 min |
| 4 | Interpretacja i błędy | 15 min |
| 5 | Modelowanie tematów | 25 min |
| 6 | Podobieństwo dokumentów i wyszukiwarka | 10 min |
| 7 | Embeddingi zdań | 15 min |
| 8 | Podsumowanie | 5 min |
| | Razem | 135 min |

## Przed zajęciami

Zajęcia wymagają Google Colab z domyślnym runtime CPU (GPU nie jest potrzebne). Pierwsza komórka szablonu instaluje sentence-transformers (`%pip install -q sentence-transformers`), pozostałe biblioteki (scikit-learn, pandas, matplotlib, NLTK) są w Colabie domyślnie. Szablon to plik lab05_szablon.ipynb, który należy wgrać do Colaba i zapisać na własnym koncie. Dane pobierają się programowo, nic nie trzeba wgrywać.

Zbiór PolEmo 2.0 jest czytany bezpośrednio z adresu

```
BASE = "https://huggingface.co/datasets/allegro/klej-polemo2-in/resolve/main/"
```

z plików train.csv, valid.csv i test.csv, każdy z kolumnami sentence (tekst) i target (etykieta). Źródło: korpus PolEmo 2.0 z repozytorium CLARIN-PL (https://clarin-pl.eu/dspace/handle/11321/710), wariant "in" z benchmarku KLEJ (https://aclanthology.org/2020.acl-main.111/), publikacja https://aclanthology.org/K19-1092/. Metadane karty zbioru w Hugging Face podają licencję CC BY-SA 4.0, ale opis w tej samej karcie wspomina CC BY-NC-SA 4.0. Dla zajęć wystarczy wiedzieć o tej rozbieżności, ale przed jakimkolwiek innym użyciem zbioru trzeba sprawdzić licencję w źródle. Kolumna target ma cztery wartości, opisane w zadaniu 1.

Wprowadzenie z opisem pojęć znajduje się w pliku lab05_wprowadzenie.md. Wykonanie wszystkich zadań trwa kilka minut, z czego najdłużej kodowanie zdań w zadaniu 7. Przy pierwszym uruchomieniu model z zadania 7 jest pobierany z internetu (około 460 MB), co wydłuża wczytanie.

W szablonie ustawione są `RANDOM_STATE = 42` oraz stałe BASE i TOKEN.

## Zadanie 1: Wczytanie i eksploracja korpusu (15 min)

1.1 Wczytaj trzy pliki CSV. Wypisz rozmiary i unikalne wartości kolumny target. Wartości mają postać `__label__meta_minus_m` (opinia negatywna), `__label__meta_plus_m` (pozytywna), `__label__meta_zero` (neutralna) i `__label__meta_amb` (niejednoznaczna, w której występują oceny przeciwnych znaków). Zweryfikuj ten opis na przykładach z punktu 1.4 i zapisz własne spostrzeżenia, zwłaszcza o klasie neutralnej.

1.2 Zamień etykiety na krótkie nazwy (neg, pos, neu, amb) w kolumnie label. Policz liczebność i udział klas w każdym z trzech zbiorów oraz liczbę duplikatów tekstów w zbiorze treningowym.

1.3 Dodaj kolumnę n_words. Opisz rozkład długości (describe), narysuj histogram i wykres pudełkowy według klasy, obejrzyj kilka najkrótszych dokumentów.

1.4 Wypisz po jednym losowym dokumencie z każdej klasy (pierwsze ok. 350 znaków). Zwróć uwagę na to, jak wyglądają teksty klasy neu i jak zapisane są odmiany (teksty są podzielone na tokeny).

1.5 Wyrażeniem regularnym podziel teksty zbioru treningowego na słowa (małe litery), policz częstości (Counter). Podaj liczbę tokenów, rozmiar słownika, udział słów występujących raz oraz 20 najczęstszych słów. Narysuj wykres ranga-częstość w skali log-log i oceń, czy rozkład przypomina prawo Zipfa.

1.6 Usuń klasę amb (jest omawiana w zadaniach dodatkowych) i przygotuj zmienne X_train, y_train, X_valid, y_valid, X_test, y_test. Zbiór testowy nie bierze udziału w żadnej decyzji aż do końca zadania 3.

Wynik: tabela liczebności klas, wykresy z 1.3 i 1.5, rozmiar słownika. Pytanie: czym klasa neu różni się od pozostałych klas poza sentymentem i jak to może wpłynąć na klasyfikator?

## Zadanie 2: Reprezentacja tekstu (20 min)

2.1 Napisz funkcje normalize(text) (małe litery, ujednolicenie białych znaków) i tokenize(text) z użyciem `re.findall`. Sprawdź je na jednym dokumencie.

2.2 Zbuduj bag of words przez CountVectorizer(token_pattern=TOKEN) dopasowany na X_train. Podaj kształt macierzy, liczbę niezerowych elementów, gęstość oraz pamięć zajmowaną przez macierz rzadką i pamięć, którą zajęłaby macierz gęsta. Wypisz najczęstsze słowa pierwszego dokumentu z licznikami.

2.3 Zbuduj TfidfVectorizer(token_pattern=TOKEN) na X_train. Wypisz słowa o najniższym i najwyższym idf oraz najwyżej ważone słowa jednego dokumentu. Sprawdź normę wiersza.

2.4 Zestaw w tabeli rozmiar słownika dla kilku ustawień wektoryzatora: min_df=1, 2, 5, max_df=0.5, ngram_range=(1, 2) oraz analyzer="char_wb" z ngram_range=(3, 5). Dla każdego podaj liczbę cech, średnią liczbę niezerowych elementów na dokument i czas dopasowania.

2.5 Sprawdź, czy SnowballStemmer z NLTK obsługuje język polski (lista `SnowballStemmer.languages`). Dla odmian słowa "lekarz" (lekarza, lekarzowi, lekarze, lekarzy, lekarką) porównaj zbiory n-gramów znakowych (char_wb, 3-5) miarą Jaccarda względem formy podstawowej.

Wynik: tabela z 2.4 i tabela podobieństw z 2.5. Pytania: ile razy mniej cech daje min_df=2 niż min_df=1 i co to oznacza dla słów występujących raz? Dlaczego n-gramy znakowe są sensowną alternatywą dla stemmingu w języku polskim i jaką cenę się za nie płaci?

## Zadanie 3: Klasyfikacja sentymentu (30 min)

3.1 Przygotuj walidację krzyżową `StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)`. Dla wektoryzacji TF-IDF (słowa 1-2, min_df=2) porównaj w CV (F1 makro, dokładność, czas) cztery modele, każdy jako `Pipeline(TfidfVectorizer, klasyfikator)`: DummyClassifier (klasa najliczniejsza), ComplementNB, LogisticRegression, LinearSVC. Wektoryzator ma być częścią potoku, żeby słownik był uczony tylko na części treningowej każdej iteracji.

3.2 Dla LinearSVC porównaj w CV warianty wektoryzacji: słowa 1 (min_df=1), słowa z max_df=0.5, słowa 1-2, słowa 1-2 z sublinear_tf, n-gramy znakowe char_wb 3-5. Zestaw F1 makro, odchylenie standardowe i czas w tabeli.

3.3 Dla najlepszego potoku dobierz parametr C (GridSearchCV na zbiorze treningowym, np. 0,1, 0,3, 1, 3, 10; F1 makro).

3.4 Dopasuj najlepszy potok na całym X_train i oceń na zbiorze walidacyjnym: classification_report i macierz pomyłek (ConfusionMatrixDisplay). Zapisz F1 makro w słowniku results.

3.5 Jednorazowo oceń wybrany model na zbiorze testowym i zapisz wynik. Nie wracaj już do strojenia.

Wynik: dwie tabele CV, raport klasyfikacji, macierz pomyłek, wynik na teście. Pytania: czy różnice między wariantami wektoryzacji przekraczają odchylenie standardowe z CV? Dlaczego baseline ma wysoką dokładność, a niską F1 makro? Czemu wynik na zbiorze testowym jest nieco inny niż na walidacyjnym?

## Zadanie 4: Interpretacja i błędy (15 min)

4.1 Z najlepszego potoku wyciągnij nazwy cech (`get_feature_names_out`) i współczynniki (`coef_`) klasyfikatora. Zestaw w tabeli 15 cech o największych współczynnikach dla każdej klasy. Sprawdź, które cechy opisują treść opinii, a które wyłącznie gatunek tekstu.

4.2 Zbuduj tabelę błędów na zbiorze walidacyjnym (tekst, etykieta, predykcja, długość). Zestaw tabelę krzyżową etykiet i predykcji, policz odsetek błędów w kwartylach długości (pd.qcut) i wypisz kilka błędów, w których model miał największy margines decision_function. Dla każdego sformułuj hipotezę, czemu model się pomylił (ironia, negacja, opinia mieszana, zastrzeżenie na końcu pozytywnej opinii).

Wynik: tabela cech, tabela błędów według długości, kilka przykładów z komentarzem. Pytanie: czy po obejrzeniu błędów byłoby sensowne zmienić wektoryzację? W jaki sposób?

## Zadanie 5: Modelowanie tematów (25 min)

5.1 Pobierz 20 Newsgroups dla kategorii rec.autos, rec.sport.hockey, sci.space, sci.med, comp.graphics i talk.politics.mideast (subset="train", remove=("headers", "footers", "quotes"), shuffle=True, random_state=RANDOM_STATE). Usuń dokumenty krótsze niż 15 słów i wypisz liczności kategorii.

5.2 Przygotuj listę stop words: angielską z scikit-learn rozszerzoną o słowa ogólne, które po obejrzeniu pierwszych tematów dominują bez wnoszenia treści. Zbuduj TfidfVectorizer (dla NMF) i CountVectorizer (dla LDA) z min_df i max_df oraz tokenami składającymi się z liter.

5.3 Napisz funkcję top_words(components, feature_names, n) zwracającą n najważniejszych słów każdego tematu. Dopasuj NMF z K=6 na TF-IDF i LatentDirichletAllocation z K=6 na licznikach, wypisz tematy obu metod.

5.4 Zestaw tabelę krzyżową dominującego tematu dokumentu i prawdziwej kategorii dla obu modeli. Policz ARI i NMI (adjusted_rand_score, normalized_mutual_info_score).

5.5 Dobór liczby tematów. Dla NMF i k z listy (np. 3, 4, 5, 6, 8, 10, 12) policz błąd rekonstrukcji, ARI i NMI. Dla LDA podziel dokumenty na treningowe i odłożone i dla kilku wartości k policz perplexity na odłożonych (LDA jest wolniejsza, wystarczą 2 do 3 wartości, np. 4, 6, 8). Narysuj wykresy zależności od k.

5.6 Wybierz k, uzasadniając wybór, i wypisz tematy oraz tabelę krzyżową z kategoriami dla tego k.

Wynik: tematy obu metod, ARI i NMI, wykresy z 5.5. Pytania: czy któryś z tematów NMF opisuje podpis lub zbiór słów niezwiązanych z żadną dyskusją? Czy wybór k na podstawie błędu rekonstrukcji lub perplexity zgadza się z oceną tematów przez człowieka? Czemu nie spodziewamy się ARI równego 1?

## Zadanie 6: Podobieństwo dokumentów i wyszukiwarka (10 min)

6.1 Zbuduj indeks TF-IDF całego zbioru treningowego PolEmo w dwóch wariantach: słowa (min_df=2, sublinear_tf) i n-gramy znakowe (char_wb 3-5, min_df=3). Napisz funkcję search(query, vec, index, k) zwracającą k najbliższych dokumentów (cosine_similarity). Sprawdź dwa własne zapytania w obu wariantach i obejrzyj etykiety zwróconych dokumentów.

6.2 Zadaj zapytanie z formą fleksyjną, która rzadko występuje w korpusie (np. "kardiologowi zawdzięczam zdrowie"), i porównaj wyniki obu wariantów. Sprawdź, których słów zapytania nie ma w słowniku. Znajdź też dokumenty najbardziej podobne do wybranego dokumentu z korpusu.

Wynik: wyniki wyszukiwania dla kilku zapytań. Pytanie: czemu podobieństwa kosinusowe są tak niskie, choć wynik wygląda sensownie?

## Zadanie 7: Embeddingi zdań (15 min)

7.1 Wylosuj stratyfikowaną podpróbę 800 dokumentów treningowych i 300 walidacyjnych. Wczytaj `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")` i zakoduj teksty, mierząc czas. Przy pierwszym uruchomieniu model jest pobierany, więc czas wczytania będzie dłuższy. Ustaw `max_seq_length = 128`, żeby kodowanie na CPU było szybkie.

7.2 Wytrenuj LogisticRegression na embeddingach i porównaj F1 makro na podpróbie walidacyjnej z TF-IDF + LogisticRegression oraz TF-IDF + LinearSVC uczonymi na tych samych 800 dokumentach.

Wynik: tabela trzech wyników i czas kodowania. Pytania: jak na wynik wpływa skrócenie dokumentów do 128 tokenów? Czy porównanie byłoby uczciwe, gdyby embeddingi uczono na podpróbie, a TF-IDF na całym zbiorze?

## Zadanie 8: Podsumowanie (5 min)

Zbierz w jednej tabeli wyniki F1 makro z zadań 3 i 7 (słownik results). Napisz kilka zdań: który model wybrałbyś do wdrożenia, co jest dużą słabością wybranego podejścia i jaki kolejny krok podjąłbyś, mając więcej czasu.

## Pytania kontrolne

1. Co się zmienia w macierzy dokument-term po zastosowaniu TF-IDF i dlaczego wiersze są normalizowane?
2. Dlaczego wektoryzator musi być dopasowywany tylko na zbiorze treningowym i jak zapewnia to Pipeline w walidacji krzyżowej?
3. Dlaczego dla języka polskiego n-gramy znakowe mogą zastąpić stemming, ale w tych danych nie poprawiły wyniku względem słów?
4. Dlaczego do porównania modeli na niezrównoważonych klasach używa się F1 makro, a nie dokładności?
5. Czym różni się model tematyczny NMF od LDA pod względem wejścia i założeń?
6. Czemu spadek błędu rekonstrukcji wraz z k nie wystarcza do wyboru liczby tematów?
7. Kiedy embeddingi zdań są lepszym wyborem od TF-IDF, a kiedy nie warto ich używać?

## Do oddania

Notebook z zadań 1 do 8, uruchomiony od początku do końca (Środowisko wykonawcze, Uruchom wszystko), z zapisanymi wynikami i wykresami. Pod wynikami każdego zadania krótka odpowiedź na pytanie z jego końca, a na końcu notebooka odpowiedzi na pytania kontrolne. Plik należy zapisać w formacie .ipynb (Plik, Pobierz jako .ipynb) i przesłać zgodnie z ustaleniami prowadzącego.
