# Automatic Google Play Comments Analytics using Machine Learning
Automatic Machine Learning Analytics for Google Play Comments

# How to use
Import your glove model or train using your corpus with:

python generate_glove.py -c all_comments.txt -t 100 -f 100 -p 10 

generate predict model using column 3 (category A)

python analyze_glove.py ../data/BradescoCartoes.csv  3

mv output.csv output_p.csv

generate training model

../data/generateCSV.sh

python analyze_glove.py input.csv 3

mv output.csv output_m.csv

Classify
python classifier.py ../data/BradescoCartoes.csv 

# The Category

## Category A 
The meaning according business issue

+ 1 Suggestion - improvements, criticism with root case
+ 2 Complaint - criticism with no root case
+ 3 Help - difficulties, business issue
+ 4 Compliment

## Category B
Feelling of the user

+ 1 Hapiness
+ 2 Sadness/Disapointment/Fear
+ 3 Angry/Repulsion
+ 4 Confusion
+ 5 Indifferent


# The dataset

The dataset is composed in portuguese comments in GooglePlay for banking applications

| Aplication  | Filename | # of comments      |  Classified category A | Classified Category B |
|-------------|:--------:|:------------------:|:----------------------:|:---------------------:|
| Bradesco Cartões | BradescoCartoes.csv |  21040 | 145 | 48 | |
| NuBank | nuBank.csv |   10000   |   68 | 33 |
| Cartões Caixa | CartoesCaixa.csv | 8560 | 73 | 47 |
| Santander Way | SantanderWay.csv | 14240 | 63 | 0 |
| Pag Seguro | PagSeguro.csv | 46560 | 20 | 29 | 
| OuroCard | OuroCard.csv |    10318 | 63 | 0 | 
| Next Banco | Next.csv | 35598 | 60 | 0 | 
| Neon Banco | Neon.csv | 20999 | 16 | 17 | 
| ItauCard | ItauCard.csv  | 28120 | 62 | 38 | 
| Itau Banco | Itau.csv | 20798 | 63 | 0 | 
| Inter Banco | Inter.csv | 23440 | 74 | 0 | 
| HiperCard | HiperCard.csv | 12519 | 14 | 0 |
| Digio | Digio.csv | 14440 | 0 | 0 |
| Cartões CasasBahia | CasasBahia.csv | 3720 | 16 | 0 |
| Cartões Luiza | CartoesLuiza.csv | 10839 | 16 | 0 |
| Cartões Carrefour | CartoesCarrefour.csv | 6920 | 20 | 0 | 
| Bradesco NetEmpresa | BradescoNetEmpresa.csv | 12227| 0 | 0 |
| Bradesco Exclusive | BradescoExclusive.csv | 16560| 0 | 0 |
| Bradesco DIN | BradescoDIN.csv |  855| 0 | 0 |
