"""expresskurier URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from kurier import views

router = routers.DefaultRouter()
router.register(r'accounts', views.AccountViewSet)
router.register(r'products', views.ProductViewSet)
router.register(r'deliveries', views.DeliveryViewSet)
router.register(r'sessions', views.SessionViewSet)
#router.register(r'predict', views.predict)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(router.urls)),
    path('predict/', views.predict, name='predict'),
]
