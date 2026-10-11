# Uczenie maszynowe: laboratoria w Google Colab

Materiały do laboratoriów z uczenia maszynowego: sześć laboratoriów i projekt zaliczeniowy. Wszystko wykonujecie w Google Colab, na darmowym środowisku.

## Plan zajęć

| Lab | Temat | Zbiór danych |
|---|---|---|
| 1 | Proces uczenia maszynowego, przygotowanie danych, `Pipeline` | Adult |
| 2 | Klasyfikacja, metryki, próg decyzyjny, koszt błędów | German Credit |
| 3 | Regresja, regularyzacja, zespoły, strojenie, interpretacja | Ames Housing |
| 4 | Uczenie nienadzorowane | Wholesale customers, digits |
| 5 | Przetwarzanie tekstu | PolEmo 2.0, 20 Newsgroups |
| 6 | Sieci neuronowe | Fashion-MNIST, CIFAR-10 |

Projekt zaliczeniowy opisuje plik `projekt_zaliczeniowy/projekt_opis.md`.

## Zasady zaliczenia

Na zaliczenie składają się:

- obecność na zajęciach i realizacja zadań podstawowych w trakcie zajęć,
- projekt końcowy, którego prezentacja odbywa się na ostatnich zajęciach.

Wymagana jest obecność na co najmniej 70% zajęć.

Zadania dodatkowe (`labNN_rozszerzone.md`) dają możliwość podniesienia oceny końcowej. Nie zastępują one zadań podstawowych realizowanych na zajęciach ani projektu końcowego.

## Jak pracować z materiałami

Przed każdymi zajęciami przeczytajcie `labNN_wprowadzenie.md`. Na zajęciach pracujecie według `labNN_instrukcja.md` w notebooku `labNN_szablon.ipynb`. Plik `labNN_rozszerzone.md` zawiera dodatkowe zadania do wyboru.

Aby otworzyć notebook w Colabie, wybierzcie Plik, Prześlij notebook albo otwórzcie go z GitHuba. Zapiszcie własną kopię na Dysku Google, żeby nie stracić zmian. Dane pobierają się z internetu przy pierwszym uruchomieniu komórek, więc potrzebne jest połączenie. Biblioteki, których nie ma w Colabie domyślnie, instalują się komórką `%pip install` w notebooku.

Uczenie transferowe w laboratorium 6 działa szybciej na GPU (Środowisko uruchomieniowe, Zmień typ środowiska uruchomieniowego), ale ukończycie je też na procesorze.

## Literatura

1. A. Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, wyd. 3, O'Reilly, 2022.
2. W. McKinney, Python for Data Analysis, wyd. 3, O'Reilly, 2022.
3. I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, 2016.
4. F. Chollet, Deep Learning with Python, wyd. 2, Manning, 2021.
5. A. Zhang, Z. Lipton, M. Li, A. Smola, Dive into Deep Learning, https://d2l.ai.
6. Dokumentacja scikit-learn: https://scikit-learn.org/stable/user_guide.html.
