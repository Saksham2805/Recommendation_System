"""Glue code to run the existing scraping + enrichment + FAISS history pipeline.

This module is deliberately thin: it just orchestrates the scripts you
already have under ``extract_history`` and ``index_creation`` so that
calling ``run_service_sync(account)`` will:

1. Scrape Netflix or Amazon Prime viewing history using Selenium.
2. Enrich titles with genre/description using ``MovieDetailsFetcher``.
3. Rebuild the combined user history FAISS index using
   ``UserHistoryIndexer``.

NOTE: For this local demo, we continue to use the hard-coded
"saksham" user id used by the original project for the history index
and JSON filenames. This keeps the existing ``HybridRecommender``
working without changes – the latest scraped history simply replaces
those files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from django.conf import settings

from extract_history.extract_netflix import get_netflix_full_history
from extract_history.extract_prime import get_amazon_prime_history
from extract_history.get_movie_details import MovieDetailsFetcher
from index_creation.create_user_history_index import UserHistoryIndexer
from .models import StreamingAccount
from .crypto_utils import decrypt_password


# These constants align with how ``HybridRecommender.load_user_history``
# currently looks up the enriched history JSON files.
USER_ID = "saksham"  # keep existing convention for now
HISTORIES_DIR = Path("histories")
DETAILED_DIR = HISTORIES_DIR / "detailed"


def _ensure_dirs() -> None:
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    DETAILED_DIR.mkdir(parents=True, exist_ok=True)


def _build_enriched_history(csv_path: Path, output_json: Path, fetcher: MovieDetailsFetcher) -> None:
    """Run LLM enrichment for a single CSV -> JSON file."""

    fetcher.process_csv_and_save(str(csv_path), str(output_json))


def _rebuild_combined_history_index() -> None:
    """Rebuild the combined FAISS user-history index.

    This reuses ``UserHistoryIndexer`` and writes files to the same
    saksham-specific directory the original project uses
    (``faiss_indexes/saksham``).
    """

    indexer = UserHistoryIndexer()

    # Collect all enriched JSON files under histories/detailed
    json_files: List[str] = []
    for name in os.listdir(DETAILED_DIR):
        if name.endswith("_enriched.json"):
            json_files.append(str(DETAILED_DIR / name))

    if not json_files:
        return

    indexer.create_combined_index(json_files=json_files, index_name="combined_history")


def run_service_sync(account: StreamingAccount) -> None:
    """Run the full pipeline for a single streaming account.

    Depending on ``account.service.slug`` this will scrape Netflix or
    Amazon Prime history, enrich titles, and rebuild the combined user
    history FAISS index.

    Any exceptions should be handled by the caller (``connect_and_sync``).
    """

    _ensure_dirs()

    email = account.username_or_email
    password = decrypt_password(account.encrypted_password)
    profile_name = account.profile_name or "Profile"

    # We'll always save the *enriched* JSON using the canonical
    # saksham-based filenames so that HybridRecommender.load_user_history
    # continues to work without changes.
    fetcher = MovieDetailsFetcher()

    if account.service.slug == "netflix":
        # Scrape Netflix history to a CSV
        csv_df = get_netflix_full_history(email=email, password=password, profile_name=profile_name, output_dir=str(HISTORIES_DIR))
        # The scraper already writes a CSV file; we just need its path.
        # It saves as netflix_history_<safe_profile>.csv – we derive that.
        safe_profile = profile_name.replace(" ", "_").lower()
        csv_path = HISTORIES_DIR / f"netflix_history_{safe_profile}.csv"

        # Enrich and save under canonical saksham filename
        output_json = DETAILED_DIR / f"netflix_history_{USER_ID}_enriched.json"
        _build_enriched_history(csv_path, output_json, fetcher)

    elif account.service.slug == "amazon_prime":
        # Scrape Prime history
        safe_profile = profile_name.replace(" ", "_").lower()
        csv_path = HISTORIES_DIR / f"prime_history_{safe_profile}.csv"
        # The scraper lets us specify the output filename directly
        get_amazon_prime_history(email=email, password=password, profile_name=profile_name, output_file=str(csv_path))

        output_json = DETAILED_DIR / f"prime_history_{USER_ID}_enriched.json"
        _build_enriched_history(csv_path, output_json, fetcher)

    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported service slug: {account.service.slug}")

    # Rebuild combined history index from whatever enriched files exist
    _rebuild_combined_history_index()
