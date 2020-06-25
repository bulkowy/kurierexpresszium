for c in 'Warszawa' 'Police' 'Mielec' 'Szczecin' 'Radom' 'Kutno' 'Gdynia' 'Konin'
do
	for d in $(seq 0 1 6)
	do
		for h in $(seq 1 6 24)
		do
			curl --header "Content-Type: application/json" --request POST --data '{"city":"'$c'", "shipment_day":'$d', "hour": '$h'}' http://localhost:8000/predict/ 
			echo -e "\n"
		done
	done
done
