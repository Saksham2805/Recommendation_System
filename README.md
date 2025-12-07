# StreamSense – Hybrid Cross‑Platform Streaming Recommender

This repository contains **StreamSense**, a hybrid recommendation system plus a small web app that:

- Scrapes and enriches your **Netflix** and **Amazon Prime** viewing history.
- Builds **FAISS** vector indexes over both catalog data and your personal history.
- Exposes a **Django REST API** that runs a hybrid (keyword + semantic) recommender implemented in `recommender.py`.
- Provides a **React (Vite + TypeScript) frontend** for interactive recommendations and connecting your streaming accounts.

---

## Quickstart (Local Demo)

This is the fastest way to run the **Django + React** app locally.

> Tested with **Python 3.10+** and **Node 18+**.

1. **Clone the repo & create a virtualenv**

   ```bash
   git clone https://github.com/Saksham2805/Recommendation_System.git
   cd Recommendation_System

   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # macOS/Linux
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root:

   ```env
   GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"        # required for LLM-based enrichment
   STREAMING_CREDENTIAL_KEY="YOUR_FERNET_KEY"  # used to encrypt stored streaming credentials
   ```

   See [Generating a Fernet key](#generating-a-fernet-key) if you don’t have a key yet.

4. **Run the Django backend**

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

   The API will be available at http://127.0.0.1:8000/.

5. **Run the React frontend** (in a second terminal):

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Open the Vite dev URL from the terminal (usually http://localhost:5173/).

6. **Use the app**

   - **Discover** tab → run hybrid recommendations.
   - **Connections** tab → (optional) connect Netflix / Prime for local history sync.

---

## High‑Level Architecture

The project is organized into three main layers:

1. **Core recommendation engine (Python scripts in repo root)**
2. **Data ingestion & enrichment utilities (`extract_history`, `histories`, `index_creation`, `data_cleaning`)**
3. **Web application (Django backend + React frontend)**

### 1. Core Hybrid Recommender

The heart of the system is `recommender.py`:

- Uses a **sentence‑transformers** model (`all-MiniLM-L6-v2`) to embed movie/show metadata.
- Uses **FAISS** for fast similarity search over those embeddings.
- Uses a custom **TF‑IDF keyword engine** (`keyword_search.py`) for lexical matching.
- Uses an LLM‑powered feature extractor (`llm_feature_extractor.py`) to parse natural language queries.
- Combines keyword + semantic scores with dynamic weighting and additional boosts (genre, actors, era, recency) to rank results.
- Can optionally build a **user taste profile** from history embeddings to personalize semantic search.

Key files:

- `recommender.py` – CLI interface + `HybridRecommender` class implementing the hybrid logic.
- `llm_feature_extractor.py` – wraps Google Gemini (via `google-generativeai`) to turn free‑form text into structured features.
- `keyword_search.py` – builds TF‑IDF matrices and runs keyword search + filtering.

### 2. Data Ingestion, Enrichment & Indexing

This layer handles cleaning source data, scraping streaming history, enriching it, and building FAISS indexes.

#### Data directories

- `data/raw/` – raw platform catalog data (CSVs).
- `data/cleaned/` – cleaned/standardized catalog data.
- `histories/` – user viewing histories (CSV + JSON):
  - `histories/detailed/` – enriched JSON histories, e.g.
    - `netflix_history_saksham_enriched.json`
    - `prime_history_saksham_enriched.json`
- `faiss_indexes/` – FAISS indexes and metadata:
  - `faiss_indexes/saksham/combined_history_index.faiss`
  - `*_index.faiss` + `*_metadata.pkl` for catalog indexes.

#### Cleaning & index creation

- `data_cleaning/prepare_data.py`
  - Cleans raw catalog data in `data/raw/`.
  - Writes cleaned CSVs into `data/cleaned/`.

- `index_creation/create_index.py`
  - Builds FAISS indexes over the cleaned catalog data.
  - For each platform, generates:
    - `<platform>_index.faiss`
    - `<platform>_metadata.pkl`

- `index_creation/create_user_history_index.py`
  - Builds a FAISS index over enriched user history JSON files.
  - Writes per‑user history indexes under `faiss_indexes/saksham/` in the current local setup.

#### Scraping & enrichment (`extract_history/`)

- `extract_history/extract_netflix.py` – uses Selenium to log in and download full Netflix viewing history for a profile.
- `extract_history/extract_prime.py` – similar for Amazon Prime Video.
- `extract_history/get_movie_details.py`
  - Uses Google Gemini to enrich a CSV of titles with `genre` and `description`.
  - Caches results in a small SQLite DB via `extract_history/database_utils.py` (`MovieDatabase`).

### 3. Web Application

The newer part of this repo is a **Django project (`backend/`)** plus a **React frontend (`frontend/`)** that wrap the core logic into APIs and a UI.

#### Django backend

- `backend/` – Django project configuration:
  - `backend/settings.py` – Django + DRF settings, installed apps, database, etc.
  - `backend/urls.py` – routes API endpoints under `/api/`.

- `recommendations/` – API over `HybridRecommender`:
  - `recommendations/views.py` – exposes an endpoint (e.g. `POST /api/recommendations/query/`) that:
    - Instantiates `HybridRecommender`.
    - Loads user history & taste profile (currently based on the `saksham` user files).
    - Runs the hybrid search and returns results + stats as JSON.
  - `recommendations/urls.py` – URL patterns for the recommendations API.

- `streaming/` – tracks Netflix/Prime accounts and runs sync:
  - `streaming/models.py` – defines:
    - `StreamingService(slug, name)` for services like `netflix`, `amazon_prime`.
    - `StreamingAccount(user, service, username_or_email, encrypted_password, profile_name, status, last_synced_at, ...)`.
  - `streaming/crypto_utils.py` – encrypts/decrypts stored passwords with **Fernet**:
    - Reads `STREAMING_CREDENTIAL_KEY` from `.env` using `python-dotenv`.
  - `streaming/sync_pipeline.py` – glue layer that:
    - Uses `extract_history.*` scrapers to pull new Netflix/Prime history.
    - Uses `MovieDetailsFetcher` to enrich those histories.
    - Rebuilds the combined user history index via `UserHistoryIndexer`.
  - `streaming/views.py` – REST endpoints:
    - `GET /api/streaming/accounts/` – list per‑user streaming connections + sync status.
    - `POST /api/streaming/connect/` – store credentials, trigger a sync via `run_service_sync`.
  - `streaming/urls.py` – URL patterns for streaming endpoints.

> Note: for now, the web API still uses a single demo user and the `saksham` history/index paths. The structure is ready to evolve into full multi‑user support.

#### React frontend (`frontend/`)

Vite + TypeScript SPA that consumes the Django API:

- `frontend/src/App.tsx`
  - **Discover** page: hybrid recommendation UI
    - Query input, platform filters (All / Netflix / Amazon Prime), number‑of‑results slider.
    - Calls `POST /api/recommendations/query/` and renders result cards with scores.
  - **Connections** page: Netflix/Prime credential management
    - Two cards: "Connect Netflix" and "Connect Prime Video".
    - Forms for username/email, password, profile name.
    - Calls `GET /api/streaming/accounts/` to show status.
    - Calls `POST /api/streaming/connect/` on "Save & Sync" to kick off scraping + indexing.

- `frontend/src/main.tsx` / `frontend/src/main.ts`
  - Vite entrypoints that mount the React app.

- `frontend/src/style.css`
  - Global theming: dark gradient background, pill buttons, card styles for results and connection forms.

- `frontend/index.html`
  - Host page for the SPA.
  - Sets the browser tab title: `StreamSense – Cross-Platform Recommender`.

---

## Environment & Configuration

Create a `.env` file in the project root with at least:

```env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
STREAMING_CREDENTIAL_KEY="YOUR_FERNET_KEY"
```

`GOOGLE_API_KEY` is used by the Gemini-based feature extractor and enrichment scripts.
`STREAMING_CREDENTIAL_KEY` is used by `streaming/crypto_utils.py` to encrypt/decrypt your
stored streaming credentials using Fernet (from `cryptography`).

### Generating a Fernet key

To generate a fresh Fernet key locally, run:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Copy the printed value into your `.env` file:

```env
STREAMING_CREDENTIAL_KEY="PASTE_GENERATED_KEY_HERE"
```

The `.env` file is loaded by:

- `extract_history/get_movie_details.py` (for `GOOGLE_API_KEY`).
- `streaming/crypto_utils.py` (for `STREAMING_CREDENTIAL_KEY`).

---

## Running the Core Recommender (CLI)

1. **Create and activate a virtualenv**

```bash
python -m venv myenv
myenv\Scripts\activate  # Windows
# or
source myenv/bin/activate  # macOS/Linux
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare data**

```bash
python data_cleaning/prepare_data.py
```

4. **Build indexes**

```bash
# Catalog indexes
python index_creation/create_index.py

# User history index
python index_creation/create_user_history_index.py
```

5. **Run the CLI recommender**

```bash
python recommender.py
```

You’ll be prompted for:

- User ID (e.g. `saksham` – must match the enriched history filenames).
- Platform (`Netflix`, `Amazon Prime`, or `all`).
- Query text (e.g. `dark sci-fi series like Black Mirror`).
- Number of recommendations.

The script prints recommendations grouped by platform with scores and stats.

---

## Running the Django + React Web App

### Backend (Django)

1. **Activate your virtualenv** and install requirements if not already done.

2. **Run migrations** (if needed):

```bash
python manage.py migrate
```

3. **Run the dev server**:

```bash
python manage.py runserver
```

The API will be available at `http://127.0.0.1:8000/`.

Key endpoints:

- `POST /api/recommendations/query/` – get hybrid recommendations.
- `GET /api/streaming/accounts/` – current streaming connections & sync status.
- `POST /api/streaming/connect/` – connect Netflix/Prime and run a sync.

### Frontend (React/Vite)

In a separate terminal:

```bash
cd frontend
npm install  # first time only
npm run dev
```

Open the Vite dev URL (usually `http://localhost:5173/`):

- Use the **Connections** tab to enter Netflix/Prime credentials and sync your history.
- Use the **Discover** tab to search for recommendations and filter by platform.

---

## Deployment Notes (Vercel + Backend)

- The React app in `frontend/` can be deployed to **Vercel** as a static SPA.
- The Django backend should be deployed to a host that supports long‑running processes and disk (Render/Railway/Fly/EC2/etc.).
- In production, the frontend should call the backend via an HTTPS base URL (e.g. `https://api.yourdomain.com/api/...`).
- For multi‑user support, you would:
  - Add real user authentication (JWT or session‑based).
  - Make streaming accounts, histories, and FAISS indexes per user (instead of hardcoded `saksham`).

---

## Repository Layout (Summary)

```text
Recommender/
├── manage.py                 # Django management script
├── recommender.py            # CLI hybrid recommender
├── llm_feature_extractor.py  # Gemini-based feature extractor
├── keyword_search.py         # TF-IDF search engine
├── requirements.txt
├── README.md
├── .gitignore
├── backend/                  # Django project (settings, URLs)
├── recommendations/          # Recommender API app
├── streaming/                # Streaming accounts + sync pipeline
├── frontend/                 # React/Vite frontend (TypeScript)
├── data/                     # Raw + cleaned platform data
├── data_cleaning/            # Scripts to clean raw data
├── extract_history/          # Scrapers + enrichment helpers
├── index_creation/           # Scripts to build FAISS indexes
├── faiss_indexes/            # Generated FAISS indexes + metadata
├── histories/                # User history CSV/JSON
└── movies_details_db/        # SQLite DB for movie metadata cache
```
