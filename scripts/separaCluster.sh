num_clusters=5
#python clustering.py sugest.csv $num_clusters > temp
for seq in $(seq 1 $num_clusters);do
	for line in $(cat temp); do
		cluster=$(echo $line | cut -d',' -f2);
		if [ $cluster == $seq ]; then
			head  -n$(echo $line | cut -d',' -f1) sugest.csv | tail -n1;
		fi
	done > "$seq".csv
done
