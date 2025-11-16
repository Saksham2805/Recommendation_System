from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .crypto_utils import encrypt_password
from .models import StreamingAccount, StreamingService


User = get_user_model()


DEMO_USERNAME = "demo_user"


def _get_or_create_demo_user() -> User:
    """Return a single demo user used for local testing.

    This keeps the API simple while you experiment locally. In a real
    deployment you would replace this with the authenticated user.
    """

    user, _ = User.objects.get_or_create(username=DEMO_USERNAME, defaults={"is_staff": False})
    return user


def _serialize_account(account: StreamingAccount) -> dict:
    return {
        "id": account.id,
        "service": account.service.slug,
        "service_name": account.service.name,
        "username_or_email": account.username_or_email,
        "profile_name": account.profile_name,
        "status": account.status,
        "last_synced_at": account.last_synced_at.isoformat() if account.last_synced_at else None,
        "last_error": account.last_error,
    }


@api_view(["GET"])
@permission_classes([AllowAny])
def list_accounts(request):
    """List streaming accounts for the demo user.

    This lets the frontend show whether Netflix/Prime are connected and
    when they were last synced.
    """

    user = _get_or_create_demo_user()
    accounts = StreamingAccount.objects.filter(user=user).select_related("service")
    data = [_serialize_account(a) for a in accounts]
    return Response(data, status=status.HTTP_200_OK)


@api_view(["POST"])
@permission_classes([AllowAny])
def connect_and_sync(request):
    """Create/update a streaming account and (optionally) trigger sync.

    Expected payload (local demo only):
    {
      "service": "netflix" | "amazon_prime",
      "username_or_email": "...",
      "password": "...",
      "profile_name": "Profile name to select",
      "run_sync": true   # default true
    }
    """

    data = request.data or {}
    service_slug = (data.get("service") or "").strip().lower()

    if service_slug not in {"netflix", "amazon_prime"}:
        return Response(
            {"detail": "'service' must be 'netflix' or 'amazon_prime'."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    username = (data.get("username_or_email") or "").strip()
    password = data.get("password") or ""
    profile_name = (data.get("profile_name") or "").strip()

    if not username or not password or not profile_name:
        return Response(
            {"detail": "'username_or_email', 'password' and 'profile_name' are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    run_sync = bool(data.get("run_sync", True))

    user = _get_or_create_demo_user()

    service_name = "Netflix" if service_slug == "netflix" else "Amazon Prime"
    service, _ = StreamingService.objects.get_or_create(slug=service_slug, defaults={"name": service_name})

    account, _ = StreamingAccount.objects.get_or_create(user=user, service=service)
    account.username_or_email = username
    account.encrypted_password = encrypt_password(password)
    account.profile_name = profile_name
    account.last_error = None
    account.status = "syncing" if run_sync else "never_synced"
    account.save()

    # Optionally trigger a sync via the existing scraping/enrichment pipeline.
    if run_sync:
        from .sync_pipeline import run_service_sync  # imported lazily to avoid circular imports

        try:
            run_service_sync(account)
            account.status = "synced"
            account.last_synced_at = timezone.now()
        except Exception as exc:  # pragma: no cover - demo diagnostics
            account.status = "error"
            account.last_error = str(exc)
        finally:
            account.save(update_fields=["status", "last_synced_at", "last_error"])

    return Response(_serialize_account(account), status=status.HTTP_200_OK)
