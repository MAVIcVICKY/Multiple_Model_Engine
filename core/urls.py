from django.urls import path
from .views import TextSearchView, ImageSearchView, HomeView

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("api/search/", TextSearchView.as_view(), name="text-search"),
    path("api/image-search/", ImageSearchView.as_view(), name="image-search"),
]
