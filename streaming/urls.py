from django.urls import path

from . import views

urlpatterns = [
    path("accounts/", views.list_accounts, name="list_streaming_accounts"),
    path("connect/", views.connect_and_sync, name="connect_and_sync_streaming"),
]
