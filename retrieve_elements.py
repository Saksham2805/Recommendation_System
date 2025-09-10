import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class PersonalizedRecommender:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the personalized recommender system.

        Args:
            model_name: Sentence transformer model for embeddings
        """
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Load FAISS indexes and metadata
        self.indexes = {}
        self.metadata = {}
        self._load_indexes()

    def _load_indexes(self, index_dir: str = 'faiss_indexes'):
        """Load FAISS indexes and metadata for all platforms."""
        if not os.path.exists(index_dir):
            print(f"Warning: {index_dir} directory not found!")
            return

        platform_mappings = {
            'netflix_index.faiss': 'Netflix',
            'amazon_prime_index.faiss': 'Amazon Prime',
            'disney_index.faiss': 'Disney Plus'
        }

        for file in os.listdir(index_dir):
            if file.endswith('_index.faiss'):
                platform_name = platform_mappings.get(file, file.replace('_index.faiss', ''))

                # Load FAISS index
                index_path = os.path.join(index_dir, file)
                self.indexes[platform_name] = faiss.read_index(index_path)

                # Load metadata
                metadata_file = file.replace('_index.faiss', '_metadata.pkl')
                metadata_path = os.path.join(index_dir, metadata_file)
                with open(metadata_path, 'rb') as f:
                    self.metadata[platform_name] = pickle.load(f)

                print(f"Loaded {platform_name} index and metadata")

    def load_user_history(self, user_id: str) -> Optional[pd.DataFrame]:
        """
        Load user's viewing history.

        Args:
            user_id: User identifier (e.g., 'user_01')

        Returns:
            DataFrame with user's history or None if not found
        """
        history_file = f"user_histories/{user_id}_history.csv"

        if not os.path.exists(history_file):
            print(f"User history file not found: {history_file}")
            return None

        try:
            df = pd.read_csv(history_file)
            print(f"Loaded {len(df)} entries from {user_id}'s history")
            return df
        except Exception as e:
            print(f"Error loading user history: {e}")
            return None

    def extract_user_preferences(self, user_history: pd.DataFrame) -> Dict:
        """
        Extract user preferences from viewing history.

        Args:
            user_history: DataFrame with user's viewing history

        Returns:
            Dictionary with user preferences
        """
        preferences = {
            'favorite_genres': [],
            'preferred_types': [],
            'highly_rated_content': [],
            'recent_content': [],
            'platform_preference': None
        }

        if user_history.empty:
            return preferences

        # Extract highly rated content (rating >= 4.0)
        highly_rated = user_history[user_history['user_rating'] >= 4.0]
        if not highly_rated.empty:
            preferences['highly_rated_content'] = highly_rated['title'].tolist()[:5]  # Top 5

        # Extract recent content (last 6 months)
        six_months_ago = datetime.now() - timedelta(days=180)
        recent_history = user_history.copy()
        recent_history['watch_date'] = pd.to_datetime(recent_history['watch_date'], errors='coerce')
        recent_content = recent_history[recent_history['watch_date'] >= six_months_ago]
        if not recent_content.empty:
            preferences['recent_content'] = recent_content['title'].tolist()[:5]

        # Extract favorite genres
        if 'listed_in' in user_history.columns:
            all_genres = []
            for genres in user_history['listed_in'].dropna():
                if isinstance(genres, str):
                    all_genres.extend([g.strip() for g in genres.split(',')])

            genre_counts = pd.Series(all_genres).value_counts()
            preferences['favorite_genres'] = genre_counts.head(3).index.tolist()

        # Extract preferred content types
        if 'type' in user_history.columns:
            type_counts = user_history['type'].value_counts()
            preferences['preferred_types'] = type_counts.head(2).index.tolist()

        # Platform preference
        if 'platform' in user_history.columns:
            platform_counts = user_history['platform'].value_counts()
            preferences['platform_preference'] = platform_counts.index[0]

        return preferences

    def enhance_query(self, user_query: str, preferences: Dict) -> str:
        """
        Enhance the user query with personal preferences.

        Args:
            user_query: Original user query
            preferences: User preferences dictionary

        Returns:
            Enhanced query string
        """
        enhanced_parts = [user_query]

        # Add favorite genres
        if preferences['favorite_genres']:
            genres_str = ', '.join(preferences['favorite_genres'])
            enhanced_parts.append(f"similar to genres: {genres_str}")

        # Add preferred content types
        if preferences['preferred_types']:
            types_str = ' or '.join(preferences['preferred_types'])
            enhanced_parts.append(f"preferably {types_str}")

        # Add highly rated content references
        if preferences['highly_rated_content']:
            liked_content = ', '.join(preferences['highly_rated_content'][:2])
            enhanced_parts.append(f"similar to: {liked_content}")

        enhanced_query = '. '.join(enhanced_parts)
        print(f"Enhanced query: {enhanced_query}")

        return enhanced_query

    def get_recommendations(self, enhanced_query: str, platform: Optional[str] = None,
                          k: int = 5, user_history: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get personalized recommendations based on enhanced query.

        Args:
            enhanced_query: Enhanced search query
            platform: Specific platform to search (None for all)
            k: Number of recommendations to return
            user_history: User's viewing history for filtering

        Returns:
            Dictionary with recommendations
        """
        # Generate query embedding
        query_embedding = self.model.encode([enhanced_query])
        faiss.normalize_L2(query_embedding)

        results = {}
        watched_titles = set()

        # Get already watched titles for filtering
        if user_history is not None and 'title' in user_history.columns:
            watched_titles = set(user_history['title'].dropna().str.lower())

        platforms_to_search = [platform] if platform else list(self.indexes.keys())

        for plt in platforms_to_search:
            if plt not in self.indexes:
                print(f"Warning: No index found for platform {plt}")
                continue

            # Search in FAISS index
            scores, indices = self.indexes[plt].search(query_embedding, k * 2)  # Get more results for filtering

            # Get metadata for results
            platform_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid result
                    result_row = self.metadata[plt].iloc[idx]
                    title = result_row.get('title', '').lower()

                    # Skip if user has already watched this
                    if title in watched_titles:
                        continue

                    result = {
                        'score': float(score),
                        'show_id': result_row.get('show_id', ''),
                        'title': result_row.get('title', ''),
                        'type': result_row.get('type', ''),
                        'cast': result_row.get('cast', ''),
                        'duration': result_row.get('duration', ''),
                        'listed_in': result_row.get('listed_in', ''),
                        'description': result_row.get('description', ''),
                        'platform': plt,
                        'release_year': result_row.get('release_year', '')
                    }

                    platform_results.append(result)

                    # Stop when we have enough results for this platform
                    if len(platform_results) >= k:
                        break

            results[plt] = platform_results[:k]  # Limit to k results per platform

        return results

    def display_recommendations(self, results: Dict, preferences: Dict, user_query: str):
        """Display recommendations in a user-friendly format."""
        print("\n" + "="*80)
        print("🎬 PERSONALIZED RECOMMENDATIONS")
        print("="*80)
        print(f"Based on your query: '{user_query}'")
        print(f"Your preferences: {', '.join(preferences['favorite_genres'][:2])}")
        print("="*80)

        total_results = 0
        for platform, platform_results in results.items():
            if platform_results:
                print(f"\n{platform.upper()} RECOMMENDATIONS:")
                print("-" * 40)

                for i, result in enumerate(platform_results, 1):
                    print(f"{i}. 🎯 {result['title']} (Score: {result['score']:.3f})")
                    print(f"   📺 Type: {result['type']} | Year: {result['release_year']}")
                    print(f"   📝 Genres: {result['listed_in']}")
                    if result['description']:
                        desc = result['description'][:100] + "..." if len(result['description']) > 100 else result['description']
                        print(f"   📖 {desc}")
                    print()

                total_results += len(platform_results)

        if total_results == 0:
            print("❌ No recommendations found. Try adjusting your query or platform selection.")

        print("="*80)


def main():
    """Main function to run the personalized recommender."""
    print("🎬 Streaming Content Personalized Recommender")
    print("="*50)

    # Initialize recommender
    try:
        recommender = PersonalizedRecommender()
    except Exception as e:
        print(f"Error initializing recommender: {e}")
        return

    # Get user input
    try:
        user_id = input("Enter user ID (e.g., user_01): ").strip()
        if not user_id:
            print("User ID cannot be empty!")
            return

        platform_input = input("Enter platform (Netflix/Amazon Prime/Disney Plus) or 'all': ").strip()
        platform = None if platform_input.lower() == 'all' else platform_input

        user_query = input("Enter your search query: ").strip()
        if not user_query:
            print("Query cannot be empty!")
            return

        k_input = input("Number of recommendations (default 5): ").strip()
        k = int(k_input) if k_input.isdigit() else 5

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    # Load user history
    user_history = recommender.load_user_history(user_id)
    if user_history is None:
        print("Cannot proceed without user history.")
        return

    # Extract preferences
    preferences = recommender.extract_user_preferences(user_history)
    print(f"Extracted preferences: {preferences}")

    # Enhance query
    enhanced_query = recommender.enhance_query(user_query, preferences)

    # Get recommendations
    print("Searching for recommendations...")
    results = recommender.get_recommendations(enhanced_query, platform, k, user_history)

    # Display results
    recommender.display_recommendations(results, preferences, user_query)


if __name__ == "__main__":
    main()
