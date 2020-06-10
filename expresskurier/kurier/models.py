from django.db import models

class Account(models.Model):
    '''
    User Account, defines user in express kurier env
    '''

    name = models.CharField(
        verbose_name="Name of User",
        max_length=100
    )

    city = models.CharField(
        verbose_name="City of User",
        max_length=30
    )

    street = models.CharField(
        verbose_name="Street of User",
        max_length=60
    )

class Product(models.Model):
    '''
    Defines product
    '''

    product_name = models.CharField(
        verbose_name="Product Name",
        max_length=100
    )

    category_path = models.CharField(
        verbose_name="Path of categories",
        max_length=150
    )
    # to change
    price = models.DecimalField(
        max_digits=16,
        decimal_places=2
    )

class Delivery(models.Model):
    purchase_id = models.IntegerField(
        primary_key=True
    )

    purchase_timestamp = models.DateTimeField(null=True, blank=True)

    delivery_timestamp = models.DateTimeField(null=True, blank=True)

    delivery_company = models.IntegerField(
        null=True,
        blank=True,
    )

class Session(models.Model):
    '''
    Sessions of users
    '''
    session_id = models.IntegerField(
        verbose_name="Related Session ID",
    )
    timestamp = models.DateTimeField(null=True, blank=True)
    # change when no nulls
    user = models.ForeignKey(
        to=Account, 
        on_delete=models.CASCADE,
        blank=True,
        null=True,
    )

    product = models.ForeignKey(
        to=Product,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
    )

    class Events(models.TextChoices):
        VIEW_PRODUCT = 'VIEW', 'VIEW_PRODUCT'
        BUY_PRODUCT = 'BUY', 'BUY_PRODUCT'

    event_type = models.CharField(
        max_length=20,
        choices=Events.choices,
        default=Events.VIEW_PRODUCT,
    )

    offered_discount = models.IntegerField()

    purchase_id = models.ForeignKey(
        to=Delivery,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )

