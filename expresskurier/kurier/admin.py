from django.contrib import admin
from kurier import models

# Register your models here.

admin.site.register([models.Product, models.Account, models.Delivery, models.Session])