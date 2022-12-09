#SEPARATE VALUES ALREADY CLASSIFIED FOR INPUT

COLUMN=4

#$(($1))


for file in $(ls *.csv); do CLASSIFIED=$(cat $file | cut -d';' -f"$COLUMN"| grep [0-9] | wc -l); head -n$CLASSIFIED $file >> input.csv; done
#cat OuroCard.txt  | tr ';' ',' | tr '\n' ';' | tr '\t' '\n' | cut -b2- | sed 's/;[0-9][0-9]*;/;/g'  > OuroCard.csv
