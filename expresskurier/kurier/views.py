from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from kurier import serializers, models
from model import encode_city, decode_city
from rest_framework.response import Response
import json
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
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
    required_fields = set(['city'])
    firms = [360, 516, 620]
    if set(request.data.keys()) != required_fields:
        return Response(
            data={"errors":"Needed fields: 'city'"},
            status=status.HTTP_400_BAD_REQUEST
        )
    enc_city = encode_city(request.data['city'])
    path = "/home/bulkowy/ium/expresskurier/model"

    result = {}

    for firm in firms:
        clf = joblib.load(f"{path}/{firm}_best.joblib")
        result[firm] = clf.predict([enc_city])

    winner = min(result, key=result.get)

    return Response(
        data={"result": result, "winner": winner},
        status=status.HTTP_200_OK
    )

     
        


