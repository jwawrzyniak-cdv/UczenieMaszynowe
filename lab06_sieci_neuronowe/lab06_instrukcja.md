# Laboratorium 6: Sieci neuronowe

Zbiór danych: Fashion-MNIST i CIFAR-10 (`keras.datasets`), uczenie transferowe z wagami ImageNet (`keras.applications`)
Środowisko: Google Colab, zalecany GPU (T4), CPU wystarczy

## Cel

Zadania prowadzą od pojedynczego neuronu napisanego w NumPy, przez perceptron wielowarstwowy (MLP) i sieć konwolucyjną (CNN), do uczenia transferowego. Po zajęciach powinna być jasna odpowiedź na pytanie, co dokładnie robi `model.fit`, jak rozpoznać nadmierne dopasowanie na krzywych uczenia i kiedy opłaca się sięgnąć po gotowy, wstępnie wytrenowany model. Wszystkie modele trenujemy na podzbiorach danych, tak aby całość dała się wykonać także bez GPU.

## Przed zajęciami

Potrzebny jest Keras 3 oraz scikit-learn, pandas, matplotlib i Pillow. W Colabie wszystko to jest zainstalowane, więc niczego nie trzeba doinstalowywać. Zaczynamy od otwarcia szablonu `lab06_szablon.ipynb` w Colabie. Pierwsza komórka wypisuje wersję Keras i aktywny backend, a w Colabie powinien to być TensorFlow. Kod w notebooku jest napisany tak, żeby nie zależał od backendu, więc nie używamy wywołań `tf.*`.

Zalecany jest akcelerator GPU. Włącza się go w menu: Środowisko wykonawcze, Zmień typ środowiska wykonawczego, T4 GPU, a potem Zapisz. Po zmianie runtime restartuje się i trzeba uruchomić notebook od początku. Poprawność można sprawdzić poleceniem `!nvidia-smi` w osobnej komórce. Wszystkie zadania są dobrane tak, żeby dało się je zrobić także na samym CPU. Jeśli GPU jest niedostępne (limit darmowego konta), nic się nie dzieje, trening trwa po prostu dłużej, a podzbiory danych i liczby epok są już dobrane na ten przypadek.

Zbiory pobierają się automatycznie przy pierwszym wywołaniu `load_data`, a wagi MobileNetV2 przy pierwszym utworzeniu modelu z `weights="imagenet"`. Potrzebne jest więc połączenie z internetem.

Zasady pracy na tych zajęciach. Ziarno losowości ustawiamy zawsze na początku komórki z modelem: `keras.utils.set_random_seed(RANDOM_STATE)`. Zbiór testowy wykorzystujemy tylko do ostatecznej oceny, a decyzje (liczba epok, regularyzacja, wybór aktywacji) podejmujemy na zbiorze walidacyjnym. Przy każdym treningu obserwujemy obie krzywe, treningową i walidacyjną.

## Zadanie 1: Neuron i spadek gradientu od zera

Cel: zobaczyć, że trening sieci to powtarzane obliczanie gradientu i przesuwanie wag, i porównać własną pętlę z `model.fit`.

1. Wygeneruj dwie nakładające się chmury punktów funkcją `make_blobs` (400 próbek, 2 cechy). Podziel dane na treningowe i testowe ze stratyfikacją, a cechy ustandaryzuj, dopasowując skaler wyłącznie do zbioru treningowego. Narysuj zbiór treningowy.
2. Napisz funkcje `sigmoid`, `loss_fn(w, b, X, y)` (entropia krzyżowa) i `gradients(w, b, X, y)`. Gradient względem wag to `X.T @ (p - y) / n`, a względem obciążenia średnia z `p - y`. Sprawdź swój gradient przez różnice skończone: różnica między gradientem analitycznym a numerycznym powinna być bardzo mała (o wielu rzędów wielkości mniejsza niż same gradienty).
3. Napisz funkcję `train(X, y, lr, epochs)`, która startuje od zerowych wag i w każdej epoce robi jeden krok na całym zbiorze. Uruchom ją dla lr równego 0.01, 0.1, 1, 10 i 100 (200 epok) i narysuj krzywe straty na jednym wykresie w skali logarytmicznej. Zestaw w tabeli stratę końcową i najmniejszą stratę, jaka wystąpiła w trakcie.
4. Dla trzech wybranych wartości lr narysuj granicę decyzyjną (poziomica p = 0.5 na tle prawdopodobieństw). Policz dokładność treningową i testową dla lr = 1.
5. Zbuduj ten sam model w Keras: jedna warstwa `Dense(1, activation="sigmoid")` z zerową inicjalizacją wag, optymalizator `SGD(learning_rate=1.0)`, strata `binary_crossentropy`. Trenuj z `batch_size` równym liczbie próbek treningowych, 200 epok, bez tasowania. Porównaj wagi i stratę z wynikami z punktu 3.

Wnioski zapisz w komórce pod zadaniem: jak lr wpływa na przebieg straty, co dzieje się dla lr = 100 i dlaczego wagi z Keras powinny pokrywać się z Twoimi.

## Zadanie 2: MLP na Fashion-MNIST

Cel: zbudować pierwszą sieć w Keras, nauczyć się czytać krzywe uczenia i poznać podstawowe sposoby walki z nadmiernym dopasowaniem.

1. Wczytaj Fashion-MNIST, wypisz kształty tablic, liczność klas i pokaż siatkę 16 obrazów z podpisami.
2. Przeskaluj piksele do przedziału [0, 1]. Ze zbioru treningowego wydziel 10% na walidację (stratyfikacja).
3. Ustal punkty odniesienia na zbiorze walidacyjnym: dokładność przewidywania klasy większościowej oraz model liniowy w Keras (Flatten i Dense(10, softmax), 5 epok, batch 256). Zapisz liczbę parametrów modelu liniowego.
4. Zbuduj MLP: Flatten, Dense(128, relu), Dense(64, relu), Dense(10, softmax). Wypisz `model.summary()` i sprawdź ręcznie, skąd bierze się liczba parametrów w każdej warstwie. Skompiluj z optymalizatorem Adam i funkcją straty `sparse_categorical_crossentropy`, trenuj 10 epok (batch 256) i narysuj krzywe straty oraz dokładności. Zmierz czas treningu.
5. Napisz funkcję `build_mlp` z parametrami liczby neuronów, aktywacji, Dropout, L2 i optymalizatora. Porównaj po 3 epoki (z tym samym ziarnem): relu, tanh i sigmoid z Adamem, a także relu z SGD (lr 0.01) oraz SGD (lr 0.1, momentum 0.9). Wyniki zestaw w tabeli.
6. Wylosuj 10 tys. próbek treningowych i wytrenuj na nich duży MLP (512-512) przez 20 epok. Narysuj krzywe i wskaż epokę o najmniejszej stracie walidacyjnej.
7. Dla tego samego dużego modelu porównaj: brak regularyzacji, Dropout 0.3, L2 o sile 1e-3, EarlyStopping (`patience=5`, `restore_best_weights=True`) i Dropout z EarlyStopping. Zestaw w tabeli liczbę epok, najmniejszą stratę walidacyjną, najlepszą epokę oraz dokładność treningową w ostatniej epoce. Narysuj krzywe straty walidacyjnej wszystkich wariantów na jednym wykresie.
8. Wytrenuj końcowy MLP na całym zbiorze treningowym (np. 256-128 z Dropout 0.2 i EarlyStopping) i dopiero teraz oceń go jeden raz na zbiorze testowym. Zapisz wynik w słowniku `results`, z którego skorzystasz w zadaniu 3.

Wnioski: która aktywacja i który optymalizator wypadły najlepiej przy tak krótkim treningu, od której epoki duży model zaczyna się przeuczać i która metoda regularyzacji dała najlepszą stratę walidacyjną.

## Zadanie 3: Sieć konwolucyjna na Fashion-MNIST

Cel: porównać CNN z MLP pod względem dokładności, liczby parametrów i czasu oraz sprawdzić, w czym sieć się myli i co wykrywają jej filtry.

1. Dodaj do danych oś kanału, tak aby obrazy miały kształt (28, 28, 1).
2. Zbuduj CNN: Conv2D(32, 3, relu), MaxPooling2D, Conv2D(64, 3, relu), MaxPooling2D, Flatten, Dropout(0.3), Dense(10, softmax). Zanim uruchomisz `summary()`, spróbuj przewidzieć kształt po każdej warstwie (bez dopełnienia konwolucja 3x3 zmniejsza obraz o 2 piksele w każdej osi). Trenuj do 8 epok z batchem 128 i EarlyStopping (`patience=2`). Zmierz czas.
3. Oceń CNN na zbiorze testowym, dopisz wynik do `results` i zestaw w jednej tabeli MLP z CNN: liczba parametrów, liczba epok, czas treningu, czas na epokę, dokładność testowa.
4. Narysuj macierz pomyłek (`ConfusionMatrixDisplay`) i policz czułość (recall) dla każdej klasy. Wskaż najczęstszą pomyłkę.
5. Pokaż 12 błędnie sklasyfikowanych obrazów z podpisem: klasa prawdziwa, przewidziana i pewność modelu.
6. Narysuj 32 filtry pierwszej warstwy konwolucyjnej. Zbuduj pomocniczy `keras.Model` zwracający wyjścia obu warstw Conv2D i pokaż po 8 map aktywacji dla jednego obrazu testowego.

Wnioski: ile parametrów mniej ma CNN i czy dorównał MLP dokładnością, dlaczego mniejsza liczba parametrów nie oznacza krótszego czasu epoki, które klasy są mylone i co mogą mieć ze sobą wspólnego, czy w filtrach widać jakąś strukturę.

## Zadanie 4: Augmentacja danych na CIFAR-10

Cel: sprawdzić, czy sztuczne powiększenie zbioru poprawia generalizację, gdy danych jest mało.

1. Wczytaj CIFAR-10. Wylosuj podzbiór 8000 obrazów (stratyfikacja) i podziel go na 6400 treningowych oraz 1600 walidacyjnych. Zbiór testowy zostaw w całości (10 tys. obrazów). Pokaż 16 obrazów z podpisami.
2. Napisz funkcję `build_cifar_cnn(augment)` budującą CNN z trzema blokami Conv2D (32, 64, 128 filtrów, `padding="same"`) i MaxPooling2D, a następnie Flatten, Dropout(0.3) i Dense(10, softmax). Skalowanie pikseli zrealizuj warstwą `Rescaling(1/255)` wewnątrz modelu. Wytrenuj model bez augmentacji (8 epok, batch 64), narysuj krzywe i zmierz czas.
3. Zbuduj sekwencję `RandomFlip("horizontal")`, `RandomRotation(0.05)` i `RandomZoom(0.1)` i pokaż kilkanaście losowych wariantów jednego obrazu (wywołanie warstwy z `training=True`).
4. Wytrenuj ten sam model z tymi warstwami wbudowanymi na początku modelu, z identycznymi parametrami treningu.
5. Zestaw oba modele w tabeli: dokładność treningowa, walidacyjna, różnica między nimi, dokładność testowa, czas. Narysuj `val_accuracy` obu modeli na jednym wykresie.

Wskazówka techniczna: w warstwach `RandomRotation` i `RandomZoom` podaj `fill_mode="nearest"` oraz `interpolation="nearest"`. Obraz wygląda niemal tak samo, a obliczenia są wyraźnie szybsze na CPU przy niektórych backendach.

Wnioski: jak zmienia się różnica między dokładnością treningową a walidacyjną, czy augmentacja poprawiła wynik testowy i ile kosztuje czasowo.

## Zadanie 5: Uczenie transferowe

Cel: porównać CNN trenowany od zera z modelem wstępnie wytrenowanym na ImageNet, przy tej samej niewielkiej liczbie obrazów.

1. Wylosuj 4000 obrazów CIFAR-10 (3000 treningowych i 1000 walidacyjnych) oraz 2000 obrazów testowych (stratyfikacja). Przygotuj je funkcją `keras.applications.mobilenet_v2.preprocess_input` i sprawdź zakres wartości po przetworzeniu.
2. Wczytaj `MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights="imagenet")` i zamroź bazę. Zbuduj model: wejście 32x32x3, warstwa `Resizing(96, 96)`, baza wywołana z `training=False`, `GlobalAveragePooling2D`, `Dropout(0.2)`, `Dense(10, softmax)`. Wypisz liczbę wszystkich i uczonych parametrów. Trenuj głowicę przez 3 epoki (Adam, lr 1e-3, batch 64), zmierz czas i policz dokładność walidacyjną i testową.
3. Odmroź ostatnie 30 warstw bazy, skompiluj model ponownie z Adam o lr 1e-4 (po zmianie `trainable` kompilacja jest konieczna) i trenuj 2 epoki. Zmierz czas.
4. Wytrenuj od zera CNN z zadania 4 (z augmentacją) na tych samych 3000 obrazach przez 6 epok. Zestaw w tabeli trzy podejścia: CNN od zera, MobileNetV2 z zamrożoną bazą, MobileNetV2 po dostrojeniu. Podaj liczbę epok, czas oraz dokładność walidacyjną i testową.

Wnioski: ile dało uczenie transferowe, czy dostrajanie się opłaciło oraz jak zmieniłby się wynik, gdyby obrazy były bardzo odmienne od zdjęć z ImageNet (np. zdjęcia rentgenowskie).

## Zadanie 6: Zapis modelu, nowy obraz, odtwarzalność

1. Zapisz CNN z zadania 3 do pliku `.keras`, wczytaj go funkcją `keras.models.load_model` i sprawdź, czy predykcje obu modeli na 200 obrazach testowych są identyczne. Wypisz rozmiar pliku.
2. Zasymuluj nowy obraz: zapisz wybrany obraz testowy jako PNG w rozdzielczości 280x280, wczytaj go biblioteką Pillow, sprowadź do 28x28 w skali szarości i do zakresu [0, 1], po czym zrób predykcję wczytanym modelem. Wypisz trzy najbardziej prawdopodobne klasy. Jeśli zostało czasu, wgraj własny obraz (panel plików w Colabie) i sprawdź, jak model sobie z nim radzi.
3. Wytrenuj ten sam mały model dwa razy z ustawionym ziarnem i dwa razy bez niego. Porównaj końcowe straty. Zanotuj, dlaczego na GPU wyniki mogą się nieco różnić nawet przy stałym ziarnie.

## Pytania kontrolne

1. Dlaczego sieć z wieloma warstwami, ale bez nieliniowych funkcji aktywacji, jest równoważna modelowi liniowemu?
2. Krzywa straty treningowej maleje, a walidacyjnej rośnie. Jak nazywa się ten stan i co można zrobić?
3. Czym różni się epoka od kroku (iteracji) i od wielkości batcha?
4. Dlaczego CNN ma mniej parametrów niż MLP, a na obrazach potrafi mu dorównać lub go przewyższyć, i dlaczego mimo to epoka CNN trwa dłużej?
5. Dlaczego warstwy augmentacji nie zmieniają obrazów podczas ewaluacji i predykcji?
6. Po co przy dostrajaniu zmniejsza się współczynnik uczenia i dlaczego bazę wywołuje się z `training=False`?
7. Dlaczego zbiór testowy wolno wykorzystać tylko raz, na końcu?

Odpowiedzi na pytania wpisz w komórce na końcu notebooka.

## Do oddania

Notebook `lab06_szablon.ipynb` uzupełniony o kod, z zapisanymi wynikami wszystkich komórek (uruchom całość od początku przed zapisaniem), z wykresami, tabelami porównawczymi i pisemnymi wnioskami pod każdym zadaniem oraz odpowiedziami na pytania kontrolne. Notebook musi przechodzić polecenie "Uruchom wszystko" bez błędów. Plik zapisz jako `.ipynb` (Plik, Pobierz, .ipynb) albo udostępnij link do Colaba z dostępem do odczytu.
