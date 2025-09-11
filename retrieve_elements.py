import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from llm_feature_extractor import LLMFeatureExtractor
from keyword_search import KeywordSearchEngine

class HybridRecommender:
    """
    Advanced hybrid recommender system combining LLM feature extraction,
    keyword search, semantic search, and weighted ranking.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', api_key: Optional[str] = None):
        """
        Initialize the hybrid recommender system.

        Args:
            model_name: Sentence transformer model for embeddings
            api_key: Google AI API key for LLM feature extraction
        """
        print("Loading hybrid recommender system...")

        # Initialize sentence transformer for semantic search
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize LLM feature extractor
        try:
            self.feature_extractor = LLMFeatureExtractor(api_key)
            print("LLM feature extractor initialized")
        except Exception as e:
            print(f"Warning: LLM feature extractor failed to initialize: {e}")
            self.feature_extractor = None

        # Initialize keyword search engine
        self.keyword_engine = KeywordSearchEngine()

        # Load FAISS indexes and metadata
        self.indexes = {}
        self.metadata = {}
        self.combined_metadata = None
        self._load_indexes()

        # Search weights for hybrid ranking
        self.search_weights = {
            'keyword': 0.4,      # Weight for keyword search
            'semantic': 0.6      # Weight for semantic search
        }

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

        all_metadata = []

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
                    platform_metadata = pickle.load(f)
                    self.metadata[platform_name] = platform_metadata

                    # Add platform info and collect for combined metadata
                    platform_metadata['platform'] = platform_name
                    all_metadata.append(platform_metadata)

                print(f"Loaded {platform_name} index and metadata")

        # Combine all metadata for keyword search
        if all_metadata:
            self.combined_metadata = pd.concat(all_metadata, ignore_index=True)
            print(f"Combined metadata: {len(self.combined_metadata)} total items")

            # Build keyword search index
            self.keyword_engine.build_index(self.combined_metadata)

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
            preferences['highly_rated_content'] = highly_rated['title'].tolist()[:5]

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

    def enhance_query_with_history(self, user_query: str, preferences: Dict) -> str:
        """
        Enhance the user query with personal preferences and history.

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

    def extract_features_from_query(self, user_query: str) -> Dict:
        """
        Extract features from user query using LLM.

        Args:
            user_query: The user's search query

        Returns:
            Dictionary containing extracted features
        """
        if self.feature_extractor:
            try:
                features = self.feature_extractor.extract_features(user_query)
                features['original_query'] = user_query
                return features
            except Exception as e:
                print(f"LLM feature extraction failed: {e}")

        # Fallback to basic feature extraction
        return self._basic_feature_extraction(user_query)

    def _basic_feature_extraction(self, user_query: str) -> Dict:
        """
        Basic feature extraction as fallback.

        Args:
            user_query: User query string

        Returns:
            Basic extracted features
        """
        query_lower = user_query.lower()

        # Basic genre detection
        genres = []
        genre_keywords = {
            'action': ['action', 'adventure'],
            'comedy': ['comedy', 'funny'],
            'drama': ['drama', 'dramatic'],
            'horror': ['horror', 'scary'],
            'sci-fi': ['sci-fi', 'science fiction'],
            'romance': ['romance', 'romantic'],
            'thriller': ['thriller', 'suspense'],
            'documentary': ['documentary', 'doc']
        }

        for genre, keywords in genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                genres.append(genre)

        # Basic content type detection
        content_type = 'both'
        if 'movie' in query_lower or 'film' in query_lower:
            content_type = 'movies'
        elif 'tv' in query_lower or 'show' in query_lower or 'series' in query_lower:
            content_type = 'tv_shows'

        return {
            'genres': genres,
            'actors': [],
            'directors': [],
            'keywords': [],
            'content_type': content_type,
            'mood': [],
            'era': '',
            'platform': '',
            'original_query': user_query
        }

    def perform_keyword_search(self, features: Dict, k: int = 20) -> List[Dict]:
        """
        Perform keyword search using TF-IDF.

        Args:
            features: Extracted features
            k: Number of results to return

        Returns:
            List of keyword search results
        """
        try:
            results = self.keyword_engine.search(features, k=k)
            # Filter by content type if specified
            if features.get('content_type') != 'both':
                results = self.keyword_engine.filter_by_content_type(results, features['content_type'])
            return results
        except Exception as e:
            print(f"Keyword search failed: {e}")
            return []

    def perform_semantic_search(self, enhanced_query: str, platform: Optional[str] = None,
                               k: int = 20, user_history: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Perform semantic search using sentence embeddings.

        Args:
            enhanced_query: Enhanced search query
            platform: Specific platform to search
            k: Number of results to return
            user_history: User's viewing history for filtering

        Returns:
            List of semantic search results
        """
        # Generate query embedding
        query_embedding = self.model.encode([enhanced_query])
        faiss.normalize_L2(query_embedding)

        results = []
        watched_titles = set()

        # Get already watched titles for filtering
        if user_history is not None and 'title' in user_history.columns:
            watched_titles = set(user_history['title'].dropna().str.lower())

        platforms_to_search = [platform] if platform else list(self.indexes.keys())

        for plt in platforms_to_search:
            if plt not in self.indexes:
                continue

            # Search in FAISS index
            scores, indices = self.indexes[plt].search(query_embedding, k * 2)

            # Get metadata for results
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    result_row = self.metadata[plt].iloc[idx]
                    title = result_row.get('title', '').lower()

                    # Skip if user has already watched this
                    if title in watched_titles:
                        continue

                    result = {
                        'semantic_score': float(score),
                        'title': result_row.get('title', ''),
                        'type': result_row.get('type', ''),
                        'cast': result_row.get('cast', ''),
                        'duration': result_row.get('duration', ''),
                        'listed_in': result_row.get('listed_in', ''),
                        'description': result_row.get('description', ''),
                        'platform': plt,
                        'release_year': result_row.get('release_year', ''),
                        'show_id': result_row.get('show_id', ''),
                        'index': int(idx)  # For merging with keyword results
                    }
                    results.append(result)

                    if len(results) >= k:
                        break

            if len(results) >= k:
                break

        return results[:k]

    def merge_search_results(self, keyword_results: List[Dict], semantic_results: List[Dict],
                           features: Dict) -> List[Dict]:
        """
        Merge keyword and semantic search results with weighted ranking.

        Args:
            keyword_results: Results from keyword search
            semantic_results: Results from semantic search
            features: Extracted features for weighting

        Returns:
            Merged and ranked results
        """
        # Create dictionaries for easy lookup
        keyword_dict = {result['title'].lower(): result for result in keyword_results}
        semantic_dict = {result['title'].lower(): result for result in semantic_results}

        # Get all unique titles
        all_titles = set(keyword_dict.keys()) | set(semantic_dict.keys())

        merged_results = []

        for title_lower in all_titles:
            keyword_result = keyword_dict.get(title_lower)
            semantic_result = semantic_dict.get(title_lower)

            # Combine results
            if keyword_result and semantic_result:
                # Both results exist - merge them
                combined_result = self._merge_single_result(keyword_result, semantic_result)
            elif keyword_result:
                # Only keyword result
                combined_result = keyword_result.copy()
                combined_result['semantic_score'] = 0.0
            elif semantic_result:
                # Only semantic result
                combined_result = semantic_result.copy()
                combined_result['tfidf_score'] = 0.0
            else:
                continue

            # Calculate final score
            final_score = self._calculate_final_score(combined_result, features)
            combined_result['final_score'] = final_score

            merged_results.append(combined_result)

        # Sort by final score
        merged_results.sort(key=lambda x: x['final_score'], reverse=True)

        return merged_results

    def _merge_single_result(self, keyword_result: Dict, semantic_result: Dict) -> Dict:
        """
        Merge a single keyword and semantic result.

        Args:
            keyword_result: Result from keyword search
            semantic_result: Result from semantic search

        Returns:
            Merged result dictionary
        """
        # Use semantic result as base (has more complete metadata)
        merged = semantic_result.copy()

        # Add keyword search score
        merged['tfidf_score'] = keyword_result.get('tfidf_score', 0.0)

        # Ensure all fields are present
        for key, value in keyword_result.items():
            if key not in merged or not merged[key]:
                merged[key] = value

        return merged

    def _calculate_final_score(self, result: Dict, features: Dict) -> float:
        """
        Calculate final weighted score for a result.

        Args:
            result: Merged result dictionary
            features: Extracted features

        Returns:
            Final weighted score
        """
        tfidf_score = result.get('tfidf_score', 0.0)
        semantic_score = result.get('semantic_score', 0.0)

        # Apply search weights
        keyword_weight = self.search_weights['keyword']
        semantic_weight = self.search_weights['semantic']

        # Calculate base score
        final_score = (keyword_weight * tfidf_score) + (semantic_weight * semantic_score)

        # Apply feature-specific boosts
        boost_factor = self._calculate_boost_factor(result, features)
        final_score *= boost_factor

        return final_score

    def _calculate_boost_factor(self, result: Dict, features: Dict) -> float:
        """
        Calculate boost factor based on feature matches.

        Args:
            result: Result dictionary
            features: Extracted features

        Returns:
            Boost factor (1.0 = no boost, >1.0 = boost)
        """
        boost = 1.0

        # Genre boost
        if features.get('genres'):
            result_genres = result.get('listed_in', '').lower()
            for genre in features['genres']:
                if genre.lower() in result_genres:
                    boost *= 1.2  # 20% boost for genre match

        # Actor boost
        if features.get('actors'):
            result_cast = result.get('cast', '').lower()
            for actor in features['actors']:
                if actor.lower() in result_cast:
                    boost *= 1.15  # 15% boost for actor match

        # Content type boost
        result_type = result.get('type', '').lower()
        content_type = features.get('content_type', 'both')

        if content_type == 'movies' and 'movie' in result_type:
            boost *= 1.1
        elif content_type == 'tv_shows' and ('tv' in result_type or 'show' in result_type):
            boost *= 1.1

        return boost

    def get_recommendations(self, user_query: str, platform: Optional[str] = None,
                          k: int = 5, user_history: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get hybrid recommendations combining keyword and semantic search.

        Args:
            user_query: User's search query
            platform: Specific platform to search
            k: Number of recommendations to return
            user_history: User's viewing history

        Returns:
            Dictionary with recommendations and metadata
        """
        print("🔍 Performing hybrid search...")

        # Extract features from user query
        features = self.extract_features_from_query(user_query)
        print(f"📋 Extracted features: {features}")

        # Load user preferences if history available
        preferences = {}
        enhanced_query = user_query

        if user_history is not None:
            preferences = self.extract_user_preferences(user_history)
            enhanced_query = self.enhance_query_with_history(user_query, preferences)
            print(f"📊 User preferences: {preferences}")

        # Perform keyword search
        print("🔤 Performing keyword search...")
        keyword_results = self.perform_keyword_search(features, k=k*2)

        # Perform semantic search
        print("🧠 Performing semantic search...")
        semantic_results = self.perform_semantic_search(enhanced_query, platform, k=k*2, user_history=user_history)

        # Merge and rank results
        print("⚖️ Merging and ranking results...")
        merged_results = self.merge_search_results(keyword_results, semantic_results, features)

        # Get top-k results
        top_results = merged_results[:k]

        # Organize by platform
        results_by_platform = {}
        for result in top_results:
            plt = result.get('platform', 'Unknown')
            if plt not in results_by_platform:
                results_by_platform[plt] = []
            results_by_platform[plt].append(result)

        return {
            'results': results_by_platform,
            'features': features,
            'preferences': preferences,
            'search_stats': {
                'keyword_results': len(keyword_results),
                'semantic_results': len(semantic_results),
                'merged_results': len(merged_results),
                'final_results': len(top_results)
            }
        }

    def display_recommendations(self, results: Dict, user_query: str):
        """Display recommendations in a user-friendly format."""
        print("\n" + "="*80)
        print("🎬 HYBRID RECOMMENDATIONS")
        print("="*80)
        print(f"Based on your query: '{user_query}'")

        if results.get('features'):
            features = results['features']
            if features.get('genres'):
                print(f"🎭 Genres: {', '.join(features['genres'])}")
            if features.get('actors'):
                print(f"🎪 Actors: {', '.join(features['actors'])}")
            if features.get('content_type') != 'both':
                print(f"📺 Content Type: {features['content_type']}")

        print("="*80)

        total_results = 0
        results_data = results.get('results', {})

        for platform, platform_results in results_data.items():
            if platform_results:
                print(f"\n{platform.upper()} RECOMMENDATIONS:")
                print("-" * 40)

                for i, result in enumerate(platform_results, 1):
                    print(f"{i}. 🎯 {result['title']} (Score: {result['final_score']:.3f})")
                    print(f"   📺 Type: {result['type']} | Year: {result['release_year']}")
                    print(f"   📝 Genres: {result['listed_in']}")
                    if result.get('description'):
                        desc = result['description'][:100] + "..." if len(result['description']) > 100 else result['description']
                        print(f"   📖 {desc}")
                    if result.get('tfidf_score', 0) > 0:
                        print(f"   🔤 Keyword Score: {result['tfidf_score']:.3f}")
                    if result.get('semantic_score', 0) > 0:
                        print(f"   🧠 Semantic Score: {result['semantic_score']:.3f}")
                    print()

                total_results += len(platform_results)

        if total_results == 0:
            print("❌ No recommendations found. Try adjusting your query or platform selection.")

        # Display search statistics
        if results.get('search_stats'):
            stats = results['search_stats']
            print("="*80)
            print("📊 SEARCH STATISTICS:")
            print(f"🔤 Keyword results: {stats['keyword_results']}")
            print(f"🧠 Semantic results: {stats['semantic_results']}")
            print(f"⚖️ Merged results: {stats['merged_results']}")
            print(f"🏆 Final recommendations: {stats['final_results']}")

        print("="*80)


def main():
    """Main function to run the hybrid recommender."""
    print("🎬 Hybrid Streaming Content Recommender")
    print("="*50)

    # Initialize recommender
    try:
        recommender = HybridRecommender()
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

    # Get recommendations
    print("Searching for recommendations...")
    results = recommender.get_recommendations(user_query, platform, k, user_history)

    # Display results
    recommender.display_recommendations(results, user_query)


if __name__ == "__main__":
    main()
