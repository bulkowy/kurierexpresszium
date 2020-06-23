for c in 'Warszawa' 'Police' 'Mielec' 'Szczecin' 'Radom' 'Kutno' 'Gdynia' 'Konin'
do
	for d in $(seq 0 1 6)
	do
		curl --header "Content-Type: application/json" --request POST --data '{"city":"'$c'", "shipment_day":'$d'}' http://localhost:8000/predict/ 
	done
done
