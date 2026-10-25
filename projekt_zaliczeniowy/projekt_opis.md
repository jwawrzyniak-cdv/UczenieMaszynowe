# Projekt zaliczeniowy: od problemu do działającego modelu

## Założenia

Projekt jest główną podstawą oceny końcowej. Zadanie polega na przejściu całej drogi, którą ćwiczyliście na laboratoriach osobno: sformułowaniu problemu, przygotowaniu danych, zbudowaniu i rzetelnej ocenie modeli, interpretacji wyników i przedstawieniu wniosków osobie, która nie musi znać szczegółów technicznych.

Praca może być indywidualna albo w parach. W parach oczekujemy większego zakresu (patrz wymagania na ocenę 4,0 i 5,0), a oba nazwiska muszą być przy wyraźnie opisanych częściach pracy. Każdy autor odpowiada za cały kod i na prezentacji musi umieć wyjaśnić dowolny jego fragment.

Środowisko pracy to Google Colab. Wynikiem jest notebook, który przechodzi od początku do końca po wybraniu "Uruchom wszystko", bez ręcznego wgrywania plików (dane pobiera kod).

## Harmonogram

Terminy podaje prowadzący na pierwszych zajęciach. Projekt ma trzy punkty kontrolne.

| Etap | Co oddajecie | Moment |
|---|---|---|
| 1. Temat | Wybrany zbiór danych i jednostronicowy opis problemu: cel, zmienna docelowa, metryka sukcesu, koszty błędów, ryzyka | po laboratorium 2 |
| 2. Stan pośredni | Notebook z analizą danych, podziałem, potokiem przetwarzania i modelem odniesienia (baseline) wraz z pierwszym wynikiem | po laboratorium 4 |
| 3. Wersja końcowa | Notebook, krótka prezentacja (8 do 10 minut, plus pytania) i odpowiedzi na pytania o kod | ostatnie zajęcia |

Etapy 1 i 2 nie są oceniane liczbą punktów, ale ich brak obniża ocenę końcową (patrz kryteria). Służą temu, żeby wykryć problemy, zanim będzie za późno: zbiór za mały, cel źle postawiony, wyciek informacji o zmiennej docelowej.

## Wymagania

### Na ocenę 3,0

1. Zbiór danych z publicznego źródła, minimum 1000 obserwacji i 8 cech. Zbiory z listy poniżej są zatwierdzone z góry. Własny zbiór wymaga akceptacji w etapie 1.
2. Opis problemu w kategoriach mierzalnych: co przewidujemy, dla kogo, jaka metryka i dlaczego ta.
3. Analiza danych: rozmiar, typy, braki, rozkłady, korelacje, rozkład zmiennej docelowej, minimum 5 różnych wykresów z krótkim komentarzem do każdego.
4. Podział na zbiór treningowy i testowy wykonany przed jakimkolwiek dopasowaniem transformacji. Zbiór testowy wolno użyć raz, na końcu.
5. Przetwarzanie danych w `Pipeline` z `ColumnTransformer`.
6. Model odniesienia (`Dummy` albo najprostszy sensowny model) oraz co najmniej dwa modele rzeczywiste.
7. Ocena w walidacji krzyżowej, metryki dobrane do problemu, tabela porównawcza.
8. Wnioski: który model wybieracie, dlaczego, jakie są ograniczenia.

### Na ocenę 4,0 (dodatkowo)

9. Strojenie hiperparametrów (`GridSearchCV` lub `RandomizedSearchCV`) bez użycia zbioru testowego.
10. Co najmniej trzy rodziny algorytmów (np. model liniowy, zespół drzew, sieć neuronowa lub SVM).
11. Analiza błędów: gdzie model się myli i dlaczego, minimum jedna hipoteza sprawdzona liczbami.
12. Interpretacja: ważność cech (permutacyjna lub SHAP), co z niej wynika dla praktyki.
13. Uwzględnienie kosztów błędów lub próg decyzyjny wybrany z uzasadnieniem (klasyfikacja) albo analiza błędów w jednostkach biznesowych (regresja).

### Na ocenę 5,0 (dodatkowo)

14. Samodzielny element wykraczający poza laboratoria: nowa cecha z uzasadnieniem i sprawdzonym wpływem, nietypowa reprezentacja danych, porównanie z gotowym modelem wstępnie wytrenowanym, kalibracja prawdopodobieństw, przedziały predykcji, uczenie transferowe albo własna procedura walidacji.
15. Analiza stabilności: wynik dla kilku ziaren losowych lub podziałów, przedział niepewności wyniku.
16. Krótka, jednostronicowa notatka dla odbiorcy nietechnicznego (po polsku, bez żargonu).
17. Zapisany model i funkcja, która dla nowego rekordu zwraca predykcję wraz z czytelnym opisem.
18. Wyraźna część o etyce i ograniczeniach: zmienne wrażliwe, możliwe uprzedzenia, ryzyko niewłaściwego użycia modelu.

## Propozycje tematów

Wszystkie zbiory pobiera się programowo. Można zaproponować własny temat, byle dane były publiczne i miały licencję pozwalającą na użycie.

### Odejście klientów operatora telekomunikacyjnego

Zbiór `telco-customer-churn` w OpenML (identyfikator 42178, 7043 klientów, 19 cech). Klasyfikacja binarna, w której koszt utraty klienta jest inny niż koszt kampanii utrzymaniowej. Kolumna `TotalCharges` nie jest czysto liczbowa i wymaga uwagi.

### Kampania marketingowa banku (temat zarezerwowany)

Zbiór `bank-marketing` (identyfikator 1461, 45211 rozmów, 16 cech, w OpenML pod anonimowymi nazwami V1 do V16, opis w UCI). Klasyfikacja niezbalansowana. Cecha czasu rozmowy (`duration`, kolumna V12) nie jest znana przed wykonaniem rozmowy, więc jej użycie trzeba przemyśleć i omówić w pracy.

### Zachowania użytkowników sklepu internetowego

Zbiór `online-shoppers-intention` (identyfikator 42993, 12330 sesji, kolumna celu `Revenue`). Przewidywanie zakupu w sesji oraz segmentacja użytkowników metodami nienadzorowanymi.

### Wycena diamentów

Zbiór `diamonds` (identyfikator 42225, 53940 kamieni, 9 cech). Regresja z silnie skośnym rozkładem ceny i cechami porządkowymi (szlif, czystość, barwa). Można rozważyć błąd względny zamiast bezwzględnego.

### Popyt na rowery miejskie

Zbiór `Bike_Sharing_Demand` (identyfikator 42712, 17379 godzin, kolumna celu `count`). Regresja na danych uporządkowanych w czasie, więc podział musi być czasowy. Porównanie z podziałem losowym jest częścią zadania.

### Klasyfikacja zdjęć kwiatów (temat zarezerwowany)

Zbiór `flower_photos` (około 3,7 tys. zdjęć pięciu gatunków, archiwum pod adresem https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz, pobierane przez `keras.utils.get_file`). Uczenie transferowe, augmentacja, analiza pomyłek. Wymaga runtime z GPU.

### Opinie po polsku w innej domenie

PolEmo 2.0 w wariancie `klej-polemo2-out` (pliki CSV w repozytorium `allegro/klej-polemo2-out` na Hugging Face, licencja CC BY-SA 4.0). Model uczony na jednej dziedzinie opinii i oceniany na innej dobrze pokazuje przesunięcie rozkładu. Można też zająć się modelowaniem tematów.

### Wady wyrobów stalowych

Zbiór `steel-plates-fault` w wariancie z identyfikatorem 40982 (1941 płyt, 27 cech liczbowych, 7 klas usterek). Klasyfikacja wieloklasowa na małym zbiorze, wykrywanie anomalii jako dodatkowy kierunek.

### Temat własny

Dane z dane.gov.pl, GUS, Eurostat, Kaggle, UCI, OpenML lub innego źródła. Opis problemu i zbioru oddajcie w etapie 1.

## Struktura notebooka

Minimalny układ, który ułatwia ocenę (w szablonie `projekt_szablon.ipynb` jest gotowy szkielet):

1. Problem i cel, metryka sukcesu, koszty błędów
2. Dane: źródło, licencja, opis zmiennych
3. Analiza danych
4. Podział i potok przetwarzania
5. Model odniesienia i modele porównywane
6. Strojenie i wybór modelu
7. Ocena końcowa na zbiorze testowym
8. Interpretacja i analiza błędów
9. Wnioski, ograniczenia, dalsze kroki

Każda sekcja zawiera krótki komentarz opisowy. Same wyniki i wykresy bez słowa komentarza nie są wystarczające.

## Kryteria oceny

| Kryterium | Udział | Co oceniamy |
|---|---|---|
| Postawienie problemu | 10% | Mierzalny cel, metryka dobrana do problemu, jawne założenia |
| Dane i przygotowanie | 15% | Analiza danych, obsługa braków, poprawny podział, brak wycieku |
| Modelowanie | 20% | Poprawny dobór i użycie algorytmów, baseline, strojenie |
| Ocena modeli | 20% | Walidacja krzyżowa, właściwe metryki, wyniki na teście zgłoszone raz |
| Interpretacja i wnioski | 15% | Analiza błędów, uzasadnienie wyboru, ograniczenia |
| Jakość notebooka | 10% | Odtwarzalność, czytelność, komentarze, struktura |
| Prezentacja i obrona | 10% | Jasność wypowiedzi, odpowiedzi na pytania o kod |

Brak etapu 1 albo 2 obniża ocenę końcową o pół stopnia za każdy brakujący etap. Wyciek informacji ze zbioru testowego do procesu uczenia lub strojenia obniża ocenę z kryterium "Dane i przygotowanie" i "Ocena modeli" do minimum.

## Format oddania

1. Notebook `.ipynb` z zapisanymi wynikami, wgrany na platformę kursu. Do notebooka dołączcie link do kopii w Colabie z uprawnieniem do odczytu.
2. Jeśli używacie dodatkowych bibliotek, ich instalacja jest w pierwszej komórce.
3. Prezentacja: do 8 slajdów albo bezpośrednio notebook, 8 do 10 minut.
4. W parach: osobna notatka z podziałem pracy (kto za które sekcje).

## Praktyczne wskazówki

Zacznijcie od modelu odniesienia, zanim wypróbujecie cokolwiek skomplikowanego. Większość zbiorów z listy da się rozsądnie opisać prostym modelem, a różnica wobec baseline to najuczciwsza miara wartości waszej pracy.

Zbiór testowy zostaje zamknięty do ostatniego kroku. Jeśli zajrzycie do niego przed wyborem modelu, wynik przestaje być oceną, a staje się częścią strojenia.

Zapisujcie, co próbowaliście i co nie zadziałało. Opis nieudanej próby z wyjaśnieniem, dlaczego się nie powiodła, jest cenniejszy niż pominięcie jej.
