import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
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
        self.user_history_index = None
        self.user_history_metadata = None
        self._load_indexes()

        # Default search weights for hybrid ranking
        self.default_search_weights = {
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
        }

        all_metadata = []

        for file in os.listdir(index_dir):
            if not file.endswith('_index.faiss'):
                continue

            # Skip Disney Plus index entirely – we don't support it in the app anymore
            if file.startswith('disney_index'):
                continue

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

        # Load user history index
        try:
            user_index_path = f'faiss_indexes/saksham/combined_history_index.faiss'
            user_metadata_path = f'faiss_indexes/saksham/combined_history_metadata.pkl'

            if os.path.exists(user_index_path):
                self.user_history_index = faiss.read_index(user_index_path)
                print("✅ Loaded user history index")

            if os.path.exists(user_metadata_path):
                with open(user_metadata_path, 'rb') as f:
                    self.user_history_metadata = pickle.load(f)

        except Exception as e:
            print(f"❌ Failed to load user history index: {e}")

    def _get_user_taste_profile(self) -> np.ndarray:
        """
        Generate user taste profile by averaging all user history embeddings.
        This represents the user's overall viewing taste.

        Returns:
            User taste profile vector
        """
        if self.user_history_index is None or self.user_history_metadata is None:
            return np.zeros(self.embedding_dim)

        # Reconstruct embeddings from metadata
        n_vectors = self.user_history_index.ntotal
        user_embeddings = np.zeros((n_vectors, self.embedding_dim))

        for i in range(n_vectors):
            movie = self.user_history_metadata[i]
            text = f"{movie.get('title', '')} {movie.get('genre', '')} {movie.get('description', '')}"
            movie_embedding = self.model.encode([text.strip()])[0]
            user_embeddings[i] = movie_embedding

        # Advanced weighting: combine frequency (count) and recency (if available)
        counts = np.array([movie.get('count', 1) for movie in self.user_history_metadata])
        
        # Normalize counts to a 0-1 scale for weighting
        normalized_counts = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)

        # Create a recency score (if date_added is available)
        recency_scores = np.ones_like(normalized_counts) # Default to 1 if no date info
        
        if self.user_history_metadata and 'date_added' in self.user_history_metadata[0]:
            try:
                current_time = datetime.now()
                dates = [datetime.strptime(movie.get('date_added'), '%Y-%m-%d') for movie in self.user_history_metadata]
                time_diffs = np.array([(current_time - d).days for d in dates])
                
                # Apply exponential decay to time differences
                recency_scores = np.exp(-time_diffs / 365) # Decay factor of 1 year
            except (ValueError, TypeError):
                # Fallback if date format is incorrect or missing
                pass

        # Combine weights: 70% count, 30% recency
        final_weights = 0.7 * normalized_counts + 0.3 * recency_scores
        
        if final_weights.sum() > 0:
            final_weights /= final_weights.sum() # Re-normalize
        else:
            final_weights = None # Use simple average if no weights

        # Compute weighted average embedding
        user_taste_profile = np.average(user_embeddings, weights=final_weights, axis=0)

        return user_taste_profile

    def load_user_history(self, user_id: str) -> Optional[pd.DataFrame]:
        """
        Load user's viewing history from JSON files.

        Args:
            user_id: User identifier (e.g., 'saksham')

        Returns:
            DataFrame with user's history or None if not found
        """
        history_files = [
            f"histories/detailed/netflix_history_{user_id}_enriched.json",
            f"histories/detailed/prime_history_{user_id}_enriched.json"
        ]

        all_history = []

        for history_file in history_files:
            if not os.path.exists(history_file):
                print(f"User history file not found: {history_file}")
                continue

            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    all_history.extend(history_data)
            except Exception as e:
                print(f"Error loading user history from {history_file}: {e}")

        if not all_history:
            return None

        df = pd.DataFrame(all_history)
        print(f"Loaded {len(df)} entries from {user_id}'s history")
        return df


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
            'most_watched_content': [],
            'platform_preference': None
        }

        if user_history.empty:
            return preferences

        # Extract most watched content
        most_watched = user_history.sort_values(by='count', ascending=False)
        if not most_watched.empty:
            preferences['most_watched_content'] = most_watched['title'].tolist()[:5]

        # Extract favorite genres
        if 'genre' in user_history.columns:
            all_genres = []
            for genres in user_history['genre'].dropna():
                if isinstance(genres, str):
                    all_genres.extend([g.strip() for g in genres.split(',')])

            genre_counts = pd.Series(all_genres).value_counts()
            preferences['favorite_genres'] = genre_counts.head(3).index.tolist()

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

        # Add most watched content references
        if preferences['most_watched_content']:
            liked_content = ', '.join(preferences['most_watched_content'][:2])
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

    def perform_keyword_search(self, features: Dict, k: int = 20, platform: Optional[str] = None) -> List[Dict]:
        """
        Perform keyword search using TF-IDF.

        Args:
            features: Extracted features
            k: Number of results to return
            platform: Optional platform to filter results by

        Returns:
            List of keyword search results
        """
        try:
            # Pass the platform filter to the keyword engine
            results = self.keyword_engine.search(features, k=k, platform=platform)
            
            # Filter by content type if specified
            if features.get('content_type') != 'both':
                results = self.keyword_engine.filter_by_content_type(results, features['content_type'])
            
            return results
        except Exception as e:
            print(f"Keyword search failed: {e}")
            return []

    def perform_semantic_search(self, enhanced_query: str, features: Dict, platform: Optional[str] = None,
                               k: int = 20, user_history: Optional[pd.DataFrame] = None,
                               user_taste_profile: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Perform semantic search using sentence embeddings.

        Args:
            enhanced_query: Enhanced search query
            features: Extracted features from the original query for dynamic weighting
            platform: Specific platform to search
            k: Number of results to return
            user_history: User's viewing history for filtering
            user_taste_profile: The user's taste profile embedding

        Returns:
            List of semantic search results
        """
        # Generate query embedding
        query_embedding = self.model.encode([enhanced_query])[0].astype('float32').reshape(1, -1)

        # Combine with user taste profile if available
        if user_taste_profile is not None and user_taste_profile.any():
            taste_embedding = user_taste_profile.astype('float32').reshape(1, -1)
            
            # Dynamic personalization weight - MORE AGGRESSIVE reduction for specific queries
            personalization_weight = 0.3  # Default weight for taste profile
            num_specific_features = len(features.get('actors', [])) + len(features.get('directors', []))
            
            if num_specific_features > 2:
                # Very specific query - minimal taste influence
                personalization_weight = 0.1
            elif num_specific_features > 0:
                # Reduce taste profile influence for specific queries
                personalization_weight = max(0.15, 0.3 - 0.1 * num_specific_features)
            elif not features.get('genres') and not features.get('keywords'):
                # Increase taste profile influence for very broad queries
                personalization_weight = 0.5

            # Weighted average of query and taste
            personalized_embedding = (1 - personalization_weight) * query_embedding + personalization_weight * taste_embedding
        else:
            personalized_embedding = query_embedding

        faiss.normalize_L2(personalized_embedding)

        results = []
        watched_titles = set()

        # Get already watched titles for filtering
        if user_history is not None and 'title' in user_history.columns:
            watched_titles = set(user_history['title'].dropna().str.lower())

        platforms_to_search = [platform] if platform else list(self.indexes.keys())

        # FIXED: Collect results from ALL platforms before filtering
        all_platform_results = []

        for plt in platforms_to_search:
            if plt not in self.indexes:
                continue

            # Search in FAISS index - get MORE results per platform
            k_per_platform = k * 3  # Get 3x results per platform to ensure good selection
            scores, indices = self.indexes[plt].search(personalized_embedding, k_per_platform)

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
                        'release_year': result_row.get('release_year', 'N/A'),
                        'date_added': result_row.get('date_added', 'N/A'),
                        'show_id': result_row.get('show_id', ''),
                        'index': int(idx)
                    }
                    all_platform_results.append(result)

        # Sort all results by semantic score and take top k
        all_platform_results.sort(key=lambda x: x['semantic_score'], reverse=True)
        return all_platform_results[:k]

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

    def _get_dynamic_weights(self, features: Dict) -> Dict[str, float]:
        """
        Calculate dynamic search weights based on query features.

        Args:
            features: Extracted features from query

        Returns:
            Dynamic weights for keyword and semantic search
        """
        # Start with defaults
        keyword_weight = self.default_search_weights['keyword']
        semantic_weight = self.default_search_weights['semantic']

        specific_features = 0

        # IMPROVED: More balanced weights for actor queries
        if features.get('actors'):
            specific_features += len(features['actors'])
            keyword_weight += 0.2 * len(features['actors'])  # Reduced from 0.3 to 0.2
            keyword_weight = max(keyword_weight, 0.55)  # Reduced from 0.7 to 0.55

        # Boost keyword weight if specific directors are mentioned
        if features.get('directors'):
            specific_features += len(features['directors'])
            keyword_weight += 0.15 * len(features['directors'])  # Increased from 0.1

        # Normalize weights so they sum to 1
        total_weight = keyword_weight + semantic_weight
        if total_weight > 0:
            keyword_weight = keyword_weight / total_weight
            semantic_weight = semantic_weight / total_weight

        return {
            'keyword': keyword_weight,
            'semantic': semantic_weight
        }

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

        # Get dynamic weights based on features
        search_weights = self._get_dynamic_weights(features)

        keyword_weight = search_weights['keyword']
        semantic_weight = search_weights['semantic']

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
        
        # --- Genre Boost ---
        genre_matches = 0
        if features.get('genres'):
            result_genres = result.get('listed_in', '').lower()
            for genre in features['genres']:
                if genre.lower() in result_genres:
                    genre_matches += 1
        if genre_matches > 0:
            boost *= (1.3 + 0.1 * genre_matches)

        # --- Actor Boost ---
        actor_matches = 0
        if features.get('actors'):
            result_cast = result.get('cast', '').lower()
            for actor in features['actors']:
                if actor.lower() in result_cast:
                    actor_matches += 1
        if actor_matches > 0:
            boost *= (1.4 + 0.15 * actor_matches)

        # --- Content Type Boost ---
        result_type = result.get('type', '').lower()
        content_type = features.get('content_type', 'both')
        if content_type != 'both':
            if (content_type == 'movies' and 'movie' in result_type) or \
               (content_type == 'tv_shows' and ('tv' in result_type or 'show' in result_type)):
                boost *= 1.15

        # --- Keyword in Description/Title Boost ---
        keyword_matches = 0
        if features.get('keywords'):
            description = result.get('description', '').lower()
            title = result.get('title', '').lower()
            for keyword in features['keywords']:
                if keyword.lower() in title:
                    keyword_matches += 1.5
                elif keyword.lower() in description:
                    keyword_matches += 1
        if keyword_matches > 0:
            boost *= (1.1 + 0.05 * keyword_matches)

        # --- Era/Year Boost (for specific queries) ---
        era_boost_applied = False
        try:
            release_year = int(result.get('release_year', 0))
            if release_year > 0 and features.get('era'):
                era = features['era'].lower()
                if era in ['latest', 'new', 'recent']:
                    if release_year >= datetime.now().year - 3:
                        boost *= 1.5  # Strong boost for recent content
                        era_boost_applied = True
                elif 's' in era: # e.g., 90s, 2010s
                    start_year = int(era.replace('s', ''))
                    if start_year <= release_year < start_year + 10:
                        boost *= 1.4 # Boost for matching decade
                        era_boost_applied = True
                elif str(release_year) == era:
                    boost *= 1.5 # Strong boost for exact year match
                    era_boost_applied = True
        except (ValueError, TypeError):
            pass

        # --- Default Recency Boost (if no specific era was matched) ---
        if not era_boost_applied:
            try:
                release_year = int(result.get('release_year', 0))
                if release_year > 0:
                    current_year = datetime.now().year
                    year_diff = max(0, current_year - release_year)
                    
                    # Apply an exponential decay boost
                    decay_rate = 0.1
                    recency_boost = 1 + (0.4 * np.exp(-decay_rate * year_diff)) # Max 40% boost for brand new content
                    boost *= recency_boost
            except (ValueError, TypeError):
                pass

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

        # Generate user taste profile from the entire history index
        user_taste_profile = self._get_user_taste_profile()

        # Load user preferences if history available
        preferences = {}
        enhanced_query = user_query

        if user_history is not None:
            preferences = self.extract_user_preferences(user_history)
            enhanced_query = self.enhance_query_with_history(user_query, preferences)
            print(f"📊 User preferences: {preferences}")

        # Adjust search weights for "cold start" users (no history)
        if user_history is None or user_history.empty:
            print("❄️ Cold start user detected. Prioritizing keyword search.")
            self.default_search_weights['keyword'] = 0.6
            self.default_search_weights['semantic'] = 0.4
        else:
            # Reset to default for users with history
            self.default_search_weights['keyword'] = 0.4
            self.default_search_weights['semantic'] = 0.6

        # Perform keyword search
        print("🔤 Performing keyword search...")
        keyword_results = self.perform_keyword_search(features, k=k*2, platform=platform)

        # Perform semantic search, now enhanced with the user's taste profile
        print("🧠 Performing personalized semantic search...")
        semantic_results = self.perform_semantic_search(
            enhanced_query, features, platform, k=k*2, user_history=user_history, user_taste_profile=user_taste_profile
        )

        # Merge and rank results
        print("⚖️ Merging and ranking results...")
        merged_results = self.merge_search_results(keyword_results, semantic_results, features)

        # Filter by era if specified
        if features.get('era'):
            era = features['era'].lower()
            filtered_by_era = []
            for result in merged_results:
                try:
                    release_year = int(result.get('release_year', 0))
                    if release_year > 0:
                        if era in ['latest', 'new', 'recent'] and release_year >= datetime.now().year - 3:
                            filtered_by_era.append(result)
                        elif 's' in era:
                            start_year = int(era.replace('s', ''))
                            if start_year <= release_year < start_year + 10:
                                filtered_by_era.append(result)
                        elif str(release_year) == era:
                            filtered_by_era.append(result)
                except (ValueError, TypeError):
                    continue
            merged_results = filtered_by_era

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
        user_id = input("Enter user ID (e.g., saksham): ").strip()
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
