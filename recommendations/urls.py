from django.urls import path

from . import views

urlpatterns = [
    path("query/", views.query_recommendations, name="query_recommendations"),
]
