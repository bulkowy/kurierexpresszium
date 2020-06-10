from django.shortcuts import render
from rest_framework import viewsets
from kurier import serializers, models

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
