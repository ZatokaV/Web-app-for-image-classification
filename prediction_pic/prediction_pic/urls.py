from django.contrib import admin
from django.urls import path, include
'''Add a URL to urlpatterns'''

urlpatterns: list = [
    path('admin/', admin.site.urls),
    path('', include('prediction_app.urls')),
]
