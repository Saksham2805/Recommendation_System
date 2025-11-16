import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv


_FERNET: Optional[Fernet] = None


def _get_raw_key() -> bytes:
    """Load or generate the secret key used for encrypting passwords.

    For this project we read STREAMING_CREDENTIAL_KEY from the environment.
    If it doesn't exist, we raise an error with instructions. This keeps
    behaviour explicit even though the whole project is meant for local/demo use.
    """

    # Ensure .env is loaded so STREAMING_CREDENTIAL_KEY is available
    load_dotenv()

    key = os.getenv("STREAMING_CREDENTIAL_KEY")
    if not key:
        raise ImproperlyConfigured(
            "STREAMING_CREDENTIAL_KEY is not set. Generate one with "
            "`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            "and add it to your .env file."
        )
    return key.encode()


def _get_fernet() -> Fernet:
    global _FERNET
    if _FERNET is None:
        _FERNET = Fernet(_get_raw_key())
    return _FERNET


def encrypt_password(plain: str) -> bytes:
    """Encrypt a plaintext password into an opaque token (bytes)."""

    if plain is None:
        raise ValueError("Password cannot be None")

    f = _get_fernet()
    return f.encrypt(plain.encode("utf-8"))


def decrypt_password(token: bytes) -> str:
    """Decrypt a password token back to plaintext.

    If the token is invalid or cannot be decrypted, an InvalidToken error
    will be raised. Callers should handle this and surface a helpful
    message (e.g. "stored credentials are invalid").
    """

    if token is None:
        raise ValueError("Encrypted password token cannot be None")

    f = _get_fernet()
    try:
        return f.decrypt(token).decode("utf-8")
    except InvalidToken as exc:
        raise InvalidToken("Failed to decrypt stored password token") from exc
