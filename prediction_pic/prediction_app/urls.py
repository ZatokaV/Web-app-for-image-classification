from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

'''Add a URL to urlpatterns'''
urlpatterns: list = [
    path('', views.home, name='home'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
