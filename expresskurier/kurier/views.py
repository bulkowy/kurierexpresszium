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

@api_view(['POST'])
def predict(request, **kwargs):
    required_fields = set(['city', 'shipment_day'])
    firms = [360, 516, 620]
    if set(request.data.keys()) != required_fields:
        return Response(
            data={'errors':'Needed fields: "city", "shipment day"'},
            status=status.HTTP_400_BAD_REQUEST
        )
    #enc_city = encode_city(request.data['city'])
    path = '/home/bulkowy/ium/expresskurier/model'
    ohe = joblib.load(f'{path}/ohe.joblib')
    ohe2 = joblib.load(f'{path}/2_ohe.joblib')
    enc_city = np.array([request.data['city']]).reshape(1, -1)
    enc_day = np.array([int(request.data['shipment_day'])]).reshape(1, -1)
    tr_city = ohe.transform(enc_city)
    enc_city_day = pd.DataFrame(
        {'city': enc_city[0][0], 'shipment_day': enc_day[0][0]}, index=[0]
    )
    tr_city_day = ohe2.transform(enc_city_day)


    result = {}
    result2 = {}

    # try:
    for firm in firms:
        clf = joblib.load(f'{path}/{firm}_best.joblib')
        result[firm] = clf.predict(tr_city)

        clf2 = joblib.load(f'{path}/2_{firm}_best.joblib')
        result2[firm] = clf2.predict(tr_city_day)

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
    # except Exception as e:
    #     return Response(
    #         data = {
    #             'error': str(e)
    #         },
    #         status=status.HTTP_400_BAD_REQUEST
    #     )

     
        


