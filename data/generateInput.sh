for file in $(ls *.csv); do CLASSIFIED=$(cat $file | cut -d';' -f4| grep [0-9] | wc -l); head -n$CLASSIFIED $file >> input.csv; done
#cat OuroCard.txt  | tr ';' ',' | tr '\n' ';' | tr '\t' '\n' | cut -b2- | sed 's/;[0-9][0-9]*;/;/g'  > OuroCard.csv
