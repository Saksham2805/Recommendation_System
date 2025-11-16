from django.conf import settings
from django.db import models


class StreamingService(models.Model):
    """Represents an external streaming platform such as Netflix or Amazon Prime."""

    name = models.CharField(max_length=50)
    slug = models.SlugField(unique=True)

    class Meta:
        verbose_name = "Streaming service"
        verbose_name_plural = "Streaming services"

    def __str__(self) -> str:  # type: ignore[override]
        return self.name


class StreamingAccount(models.Model):
    """Stores a user's credentials/profile for a given streaming service.

    NOTE: For this project this is a local demo only. Credentials are stored
    encrypted, but you should NOT use this pattern for production use without
    a thorough security review and proper secret management.
    """

    STATUS_CHOICES = [
        ("never_synced", "Never synced"),
        ("syncing", "Syncing"),
        ("synced", "Synced"),
        ("error", "Error"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    service = models.ForeignKey(StreamingService, on_delete=models.CASCADE)

    username_or_email = models.CharField(max_length=255)
    encrypted_password = models.BinaryField()
    profile_name = models.CharField(max_length=255, blank=True, null=True)

    last_synced_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="never_synced")
    last_error = models.TextField(blank=True, null=True)

    # Optional: paths for user-specific history and FAISS indexes
    history_dir = models.CharField(max_length=255, blank=True, null=True)
    faiss_dir = models.CharField(max_length=255, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "service")
        verbose_name = "Streaming account"
        verbose_name_plural = "Streaming accounts"

    def __str__(self) -> str:  # type: ignore[override]
        return f"{self.user} - {self.service.name}"
