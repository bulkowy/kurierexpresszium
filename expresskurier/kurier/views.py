from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from kurier import serializers, models
from rest_framework.response import Response
import numpy as np
import pandas as pd
import joblib

class AccountViewSet(viewsets.ModelViewSet):
    queryset = models.Account.objects.all()
    serializer_class = serializers.AccountSerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = models.Product.objects.all()
    serializer_class = serializers.ProductSerializer

class DeliveryViewSet(viewsets.ModelViewSet):
    queryset = models.Delivery.objects.all()
    serializer_class = serializers.DeliverySerializer

class SessionViewSet(viewsets.ModelViewSet):
    queryset = models.Session.objects.all()
    serializer_class = serializers.SessionSerializer

def validate(request):
    required_fields = set(['city', 'shipment_day', 'hour'])
    cities = [
        'Police', 'Mielec', 'Szczecin', 'Warszawa',
        'Radom', 'Kutno', 'Gdynia', 'Konin'
    ]

    # check args
    if set(request.data.keys()) != required_fields:
        return Response(
            data={'errors':'Needed fields: "city", "shipment day", "hour"'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if request.data['hour'] not in range(0, 24):
        return Response(
            data={'errors':'"hour" field required to be in range [0,23]'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if request.data['shipment_day'] not in range(0, 7):
        return Response(
            data={'errors':'"shipment_day" field required to be in range [0,6]'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if request.data['city'] not in cities:
        return Response(
            data={'errors':f'"city" field required to be in {str(cities)}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    return None

def transform_data(request, path, hour_divisor, companies):
    # get OneHot Encoders
    ohe = [joblib.load(f'{path}/A_{c}_ohe.joblib') for c in companies]
    ohe2 = [joblib.load(f'{path}/B_{c}_ohe.joblib') for c in companies]

    # parse given data to be suitable for models
    enc_city = np.array([request.data['city']]).reshape(1, -1)
    tr_cities = {companies[i]:ohe[i].transform(enc_city) for i in range(len(companies))}

    enc_day = np.array([int(request.data['shipment_day'])]).reshape(1, -1)
    enc_hour = np.array([int(request.data['hour'])//hour_divisor]).reshape(1, -1)
    enc_cdh = pd.DataFrame(
        {'city': enc_city[0][0], 'shipment_day': enc_day[0][0], 'hour': enc_hour[0][0]}, index=[0]
    )
    tr_cdh = {companies[i]:ohe2[i].transform(enc_cdh) for i in range(len(companies))}

    return tr_cities, tr_cdh

@api_view(['POST'])
def predict(request, **kwargs):
    companies = [360, 516, 620]
    hour_divisor = 8
    path = f'/home/bulkowy/ium/expresskurier/model/{hour_divisor}hdivisor/'

    val_rq = validate(request)

    if val_rq:
        return val_rq

    tr_cities, tr_cdh = transform_data(request, path, hour_divisor, companies)

    result = {}
    result2 = {}

    try:
        for company in companies:
            clf = joblib.load(f'{path}/A_{company}_best.joblib')
            result[company] = clf.predict(tr_cities[company])

            clf2 = joblib.load(f'{path}/B_{company}_best.joblib')
            result2[company] = clf2.predict(tr_cdh[company])

        winner = min(result, key=result.get)
        winner2 = min(result2, key=result2.get)

        return Response(
            data = {
                'result': result, 
                'winner': winner,
                
                'result2': result2, 
                'winner2': winner2
            },
            status=status.HTTP_200_OK
        )
    except Exception as e:
        return Response(
            data = {
                'error': str(e)
            },
            status=status.HTTP_400_BAD_REQUEST
        )

     
        


