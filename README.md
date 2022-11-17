# Automatic Google Play Comments Analytics using Machine Learning
Automatic Machine Learning Analytics for Google Play Comments

# How to use

*   Import your glove model or train using your corpus with:

        python generate_glove.py -c all_comments.txt -t 100 -f 100 -p 10

Or import model from: http://nilc.icmc.usp.br/embeddings

*   generate training model using column 3 (category A)

        cd ../data
        ./generateInput.sh
        mv input.csv ../scripts
        cd ../scripts
        python analyze_glove.py input.csv 3
        mv output.csv output_m.csv
        
 *   generate predict model using column 3 (category A)

        python analyze_glove.py ../data/BradescoCartoes.csv
        mv output.csv output_p.csv

*    Classify

    python classifier.py file_to_split.csv

# The Category

## Category A 
The meaning according business issue

+ 1 Suggestion - improvements, criticism with root cause
+ 2 Complaint - criticism with no root cause
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

The dataset is composed in portuguese comments/reviews in GooglePlay for banking applications

| Aplication  | Filename | # of comments      |  Classified category A | Classified Category B |
|-------------|:--------:|:------------------:|:----------------------:|:---------------------:|
|Bradesco Cartões |	BradescoCartoes.csv |	21040 |	208 |	61 |
|NuBank	| nuBank.csv |	10000 |		100	 |	45 |	
|Cartões Caixa |	CartoesCaixa.csv |		8560	 |	103 |		69 |	
|Santander Way |	SantanderWay.csv	 |	14240	 |	100	 |	0 |	
|Pag Seguro |	PagSeguro.csv	 |	46560	 |	51	 |	51 |	
|OuroCard |	OuroCard.csv |		10318 |		104 |		0 |	
|Next Banco |	Next.csv |		35598	 |	107 |		0 |	
|Neon Banco |	Neon.csv |		20999	 |	24 |		24 |	
|ItauCard |	ItauCard.csv	 |	28120 |		116	 |	55 |	
|Itau Banco |	Itau.csv |		20798	 |	100 |		0 |	
|Inter Banco |	Inter.csv	 |	23440	 |	113 |		0 |	
|HiperCard |	HiperCard.csv	 |	12519 |		33 |		0 |	
|Cartões CasasBahia |	CasasBahia.csv	 |	3720	 |	44	 |	0 |	
|Cartões Luiza |	CartoesLuiza.csv |		10839 |		27 |		0 |	
|Cartões Carrefour |	CartoesCarrefour.csv |		6920	 |	27 |		0 |	
|Bradesco NetEmpresa |	BradescoNetEmpresa.csv |		12227 |		0 |		0 |	
|Bradesco Exclusive |	BradescoExclusive.csv	 |	16560	 |	0 |		0 |	
|Bradesco DIN| BradescoDIN.csv |		855 |		0	 |	0 |	
|Digio |	Digio.csv	 |	14440 |		0 |		0 |	
|TOTAL|	 |			317753 |		1257	 |	305 |	



