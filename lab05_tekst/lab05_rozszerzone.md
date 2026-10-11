# Laboratorium 5: zadania dodatkowe

Pięć zadań to niezależne alternatywy: każde da się zrobić samodzielnie i żadne nie korzysta z wyników innego. Wszystkie pracują na tych samych danych co główna część laboratorium (PolEmo 2.0 bez klasy amb, z wyjątkiem zadania R5, albo 20 Newsgroups). W nowym notebooku wystarczy skopiować komórkę z importami i wczytaniem danych z szablonu. Zadania R1, R2 i R4 wymagają dodatkowej instalacji: `%pip install -q sentence-transformers gensim spacy`.

## Zadanie R1: Lematyzacja polskiego tekstu

Sprawdź, czy sprowadzenie słów do form podstawowych poprawia klasyfikację sentymentu.

1. Zainstaluj spaCy i model dla polskiego: `python -m spacy download pl_core_news_sm`. W Colabie można to wywołać z notebooka przez `subprocess.run([sys.executable, "-m", "spacy", "download", "pl_core_news_sm"])`, a potem wywołać `importlib.invalidate_caches()` i wczytać model przez `spacy.load("pl_core_news_sm", disable=["parser", "ner"])`.
2. Napisz funkcję lemmatize(texts), która dla każdego dokumentu zwraca lematy połączone spacjami (tylko tokeny alfabetyczne, małe litery).
3. Lematyzacja jest wolna, więc wylosuj stratyfikowaną podpróbę 2500 dokumentów treningowych, zlematyzuj ją i zbiór walidacyjny i zmierz czas.
4. Porównaj LinearSVC na TF-IDF: tekst surowy kontra lematy (słowa 1 i 1-2) oraz n-gramy znakowe na tekście surowym. Podaj liczbę cech, F1 makro w 5-krotnej walidacji krzyżowej na podpróbie i na zbiorze walidacyjnym.
5. Obejrzyj pary (forma, lemat) dla wybranego dokumentu i wskaż błędne lematy.

Wynik: tabela porównawcza i lista błędów lematyzatora. Pytanie: czy lematyzacja pomogła i jak bardzo wynik zależy od jakości lematyzatora? Czemu odpowiedzi nie można wyciągnąć z jednego przebiegu?

## Zadanie R2: Klasteryzacja dokumentów

Porównaj grupowanie dokumentów algorytmem KMeans na trzech reprezentacjach.

1. Wybierz 6 kategorii 20 Newsgroups (te same co w zadaniu 5), pobierz zbiór treningowy z usuniętymi nagłówkami, stopkami i cytatami i wylosuj stratyfikowanie 1200 dokumentów.
2. Przygotuj trzy reprezentacje: TF-IDF (rzadka), TF-IDF po TruncatedSVD do 100 wymiarów (z normalizacją wierszy) i embeddingi zdań modelem paraphrase-multilingual-MiniLM-L12-v2 (znormalizowane).
3. Dla każdej reprezentacji uruchom KMeans (k=6, n_init=10, random_state=RANDOM_STATE) i policz ARI, NMI oraz współczynnik sylwetki względem prawdziwych kategorii.
4. Zestaw skupienia z kategoriami (tabela krzyżowa), wypisz najważniejsze słowa skupień TF-IDF i narysuj embeddingi zrzutowane na 2 wymiary, kolorując prawdziwymi kategoriami.

Wynik: tabela ARI, NMI i sylwetki, tabele krzyżowe, wykres. Pytania: dlaczego niska wartość współczynnika sylwetki nie oznacza złego dopasowania do kategorii? Które skupienie TF-IDF nie opisuje żadnego tematu i skąd się bierze?

## Zadanie R3: Naiwny klasyfikator Bayesa od zera

Zaimplementuj w NumPy wielomianowy naiwny klasyfikator Bayesa i porównaj go z MultinomialNB.

1. Napisz klasę MultinomialNBScratch z metodami fit i predict. W fit policz liczniki słów w każdej klasie, wygładzanie alpha i logarytmy prawdopodobieństw cech oraz priorów klas. W predict wybierz klasę o największej sumie logarytmów.
2. Dopasuj klasę na licznikach z CountVectorizer (PolEmo, trzy klasy) i porównaj z MultinomialNB: parametry (feature_log_prob_, class_log_prior_), predykcje i F1 makro na zbiorze walidacyjnym.
3. Sprawdź wpływ parametru alpha na F1 makro (np. 0,01, 0,03, 0,1, 0,3, 1, 3).
4. Dla najdłuższego dokumentu walidacyjnego policz iloczyn prawdopodobieństw cech bez logarytmów i pokaż, że wynik zaokrągla się do zera. Porównaj sumy logarytmów dla klas.
5. Porównaj w tabeli własny NB, MultinomialNB (liczniki i TF-IDF), ComplementNB i LinearSVC.

Wynik: tabela zgodności i F1 oraz wynik eksperymentu z iloczynem prawdopodobieństw. Pytania: czemu bez wygładzania model jest bezużyteczny dla słów niewidzianych w danej klasie? Dlaczego naiwny Bayes bywa dobrym klasyfikatorem mimo fałszywego założenia o niezależności słów?

## Zadanie R4: Word2Vec

Wytrenuj własne wektory słów na korpusie opinii i sprawdź, co z nich wynika.

1. Użyj gensim.models.word2vec.Word2Vec na tokenach z kolumny sentence zbioru train: vector_size=100, window=5, min_count=5, sg=1, epochs=10, seed=RANDOM_STATE (dla powtarzalności workers=1). Zmierz czas treningu.
2. Wypisz najbliższych sąsiadów (most_similar) kilku słów: lekarz, hotel, obsługa, polecam, tragedia, basen. Oceń ich sensowność. Zwróć uwagę na to, kto jest najbliższym sąsiadem słowa "polecam".
3. Wypróbuj analogie `most_similar(positive=..., negative=...)`, np. lekarka + pan - pani. Oceń, które działają, i wyjaśnij, czemu korpus tej wielkości ma ograniczenia.
4. Reprezentuj dokument jako średnią wektorów jego słów i wytrenuj LogisticRegression na tych wektorach. Porównaj F1 makro na walidacji z TF-IDF + LogisticRegression.

Wynik: tabela sąsiadów, wyniki analogii, porównanie F1. Pytania: czemu "polecam" i "odradzam" są blisko siebie, choć znaczą coś przeciwnego? Co jeszcze oprócz wielkości korpusu ogranicza jakość wektorów?

## Zadanie R5: Cztery klasy i klasa niejednoznaczna

Główna część laboratorium pomija klasę amb. W tym zadaniu wraca ona do gry.

1. Dopasuj Pipeline(TF-IDF słowa 1-2 z min_df=2, LinearSVC) na wszystkich czterech klasach, także z class_weight="balanced". Zestaw F1 makro, classification_report i macierz pomyłek na zbiorze walidacyjnym.
2. Przy pomocy modelu trójklasowego (neg, neu, pos) oceń pojedyncze zdania i sprawdź, w jakim odsetku dokumentów każdej klasy występują jednocześnie zdania neg i pos. Zdania wydziel np. po kropce.
3. Wypisz kilka dokumentów amb sklasyfikowanych błędnie i podaj, do jakich klas trafiły. Dla każdego oceń, czy wskazana klasa jest rozsądna.

Wynik: raport klasyfikacji, tabela odsetków dokumentów mieszanych, przykłady. Pytania: dlaczego klasa niejednoznaczna jest tak trudna? Czy odsetek dokumentów mieszanych dobrze odróżnia amb od pozostałych klas? Jak etykietowanie ludzkie może wpływać na górną granicę jakości?
