from rest_framework import serializers
from kurier import models

class AccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Account
        fields = '__all__'

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Product
        fields = '__all__'

class DeliverySerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Delivery
        fields = '__all__'

class SessionSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(queryset=models.Account.objects.all())
    product = serializers.PrimaryKeyRelatedField(queryset=models.Product.objects.all())
    purchase_id = serializers.PrimaryKeyRelatedField(queryset=models.Delivery.objects.all())

    class Meta:
        model = models.Session
        fields = '__all__'