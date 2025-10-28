# Hybrid Streaming Content Recommendation System

This project is an advanced hybrid recommendation system designed to provide personalized movie and TV show recommendations from multiple streaming platforms like Netflix, Amazon Prime, and Disney Plus. It leverages a combination of natural language processing, semantic search, and keyword-based filtering to deliver accurate and relevant suggestions.

## 🚀 Features

- **Hybrid Search Engine**: Combines TF-IDF for keyword matching and FAISS-powered semantic search for contextual understanding.
- **LLM-Powered Query Understanding**: Uses Google's Gemini model to extract detailed features (genres, actors, mood, etc.) from natural language queries.
- **Personalized Recommendations**: Creates a user "taste profile" based on their viewing history, with dynamic weighting for frequency and recency.
- **Multi-Platform Support**: Aggregates content from Netflix, Amazon Prime, and Disney Plus, providing recommendations from a vast library.
- **Dynamic Ranking**: Implements a sophisticated scoring system that dynamically adjusts weights and applies boosts based on query specificity and feature matching.
- **Cold Start Handling**: Provides sensible recommendations even for new users with no viewing history.

## ⚙️ System Architecture

The system is built on the following components:

1.  **Data Ingestion & Cleaning**: Raw CSV data from streaming platforms is cleaned and standardized.
2.  **User History Enrichment**: User viewing history is enriched with detailed metadata (genre, description) using an LLM.
3.  **Indexing Pipeline**:
    *   **Semantic Index**: Movie and TV show metadata is converted into vector embeddings using `Sentence-Transformers` and indexed in FAISS for efficient similarity search.
    *   **Keyword Index**: A TF-IDF matrix is built for fast and efficient keyword-based search.
4.  **LLM Feature Extractor**: A dedicated module that parses user queries to extract structured search criteria.
5.  **Hybrid Recommender Core**: The main engine that:
    *   Receives a user query.
    *   Extracts features using the LLM.
    *   Builds a personalized query vector by combining the user's query with their taste profile.
    *   Performs parallel searches (keyword and semantic).
    *   Merges, re-ranks, and boosts results to produce the final recommendations.

##  Quick Start Guide

Follow these steps to get the recommendation system up and running on your local machine.

### 1. Prerequisites

- Python 3.8+
- `git` for cloning the repository

### 2. Clone the Repository

```bash
git clone https://github.com/Saksham2805/Recommendation_System.git
cd Recommendation_System
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv myenv
myenv\Scripts\activate

# For macOS/Linux
python3 -m venv myenv
source myenv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Set Up API Keys

The system uses Google's Generative AI (Gemini) for feature extraction and data enrichment. You'll need an API key.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API key to this file:

    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

### 6. Prepare the Data

The raw data is expected to be in the `data/raw/` directory. The data cleaning script will process it.

- Run the data preparation script:
  ```bash
  python data_cleaning/prepare_data.py
  ```

### 7. Build the Search Indexes

You need to create the FAISS and TF-IDF indexes before running the recommender.

- **Build Platform Indexes**: This script processes the cleaned platform data and creates the FAISS indexes.
  ```bash
  python index_creation/create_index.py
  ```
- **Build User History Index**: This script processes the user's viewing history.
  ```bash
  python index_creation/create_user_history_index.py
  ```

### 8. Run the Recommender!

You are now ready to get recommendations.

- Run the main recommender script:
  ```bash
  python recommender.py
  ```
- Follow the on-screen prompts:
  1.  Enter your user ID (e.g., `saksham`).
  2.  Enter the desired platform (`Netflix`, `Amazon Prime`, `Disney Plus`, or `all`).
  3.  Enter your search query (e.g., `sci-fi movies with aliens`).
  4.  Specify the number of recommendations you want.

## 📂 Project Structure

```
Recommender/
│
├── .env                # API keys and environment variables
├── recommender.py        # Main application script
├── llm_feature_extractor.py # Extracts features from queries using LLM
├── keyword_search.py     # TF-IDF based search engine
├── requirements.txt      # Project dependencies
│
├── data/
│   ├── raw/            # Raw platform data (CSV)
│   └── cleaned/        # Cleaned platform data
│
├── data_cleaning/
│   └── prepare_data.py # Script to clean raw data
│
├── index_creation/
│   ├── create_index.py # Creates FAISS indexes for platforms
│   └── create_user_history_index.py # Creates FAISS index for user history
│
├── faiss_indexes/      # Stores the generated FAISS indexes and metadata
│
├── histories/
│   ├── raw/            # Raw user history data
│   └── detailed/       # Enriched user history data (JSON)
│
└── extract_history/
    ├── get_movie_details.py # Enriches history with genre/description
    └── ...
