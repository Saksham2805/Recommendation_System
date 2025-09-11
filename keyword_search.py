import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import re
from collections import defaultdict

class KeywordSearchEngine:
    """
    Implements keyword-based search using TF-IDF for movie/TV recommendations.
    """

    def __init__(self):
        """
        Initialize the keyword search engine.
        """
        self.vectorizer = None
        self.tfidf_matrix = None
        self.metadata_df = None
        self.feature_weights = {
            'genres': 0.4,
            'actors': 0.25,
            'directors': 0.2,
            'keywords': 0.1,
            'mood': 0.05
        }

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TF-IDF vectorization.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if pd.isna(text) or not text:
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _create_searchable_text(self, row: pd.Series) -> str:
        """
        Create searchable text from movie/TV metadata.

        Args:
            row: DataFrame row containing movie/TV metadata

        Returns:
            Combined searchable text
        """
        # Extract relevant fields
        title = self._preprocess_text(row.get('title', ''))
        cast = self._preprocess_text(row.get('cast', ''))
        description = self._preprocess_text(row.get('description', ''))
        listed_in = self._preprocess_text(row.get('listed_in', ''))
        director = self._preprocess_text(row.get('director', ''))

        # Combine with different weights by repetition
        parts = []

        # Title gets highest weight (3x)
        if title:
            parts.extend([title] * 3)

        # Description gets high weight (2x)
        if description:
            parts.extend([description] * 2)

        # Genres and cast get normal weight (1x)
        if listed_in:
            parts.append(listed_in)
        if cast:
            parts.append(cast)
        if director:
            parts.append(director)

        return ' '.join(parts)

    def build_index(self, metadata_df: pd.DataFrame):
        """
        Build TF-IDF index from metadata.

        Args:
            metadata_df: DataFrame containing movie/TV metadata
        """
        print("Building TF-IDF index for keyword search...")

        # Store metadata
        self.metadata_df = metadata_df.copy()

        # Create searchable text for each item
        self.metadata_df['searchable_text'] = self.metadata_df.apply(
            self._create_searchable_text, axis=1
        )

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size
            stop_words='english',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=1,
            max_df=0.95
        )

        # Fit and transform the searchable text
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.metadata_df['searchable_text']
        )

        print(f"TF-IDF index built with {self.tfidf_matrix.shape[0]} documents and {self.tfidf_matrix.shape[1]} features")

    def _create_query_vector(self, features: Dict) -> np.ndarray:
        """
        Create TF-IDF query vector from extracted features.

        Args:
            features: Dictionary of extracted features

        Returns:
            TF-IDF query vector
        """
        # Combine all features into a single query string
        query_parts = []

        # Add genres with weight
        if features.get('genres'):
            genre_text = ' '.join(features['genres'])
            query_parts.extend([genre_text] * 3)  # Higher weight for genres

        # Add actors with weight
        if features.get('actors'):
            actor_text = ' '.join(features['actors'])
            query_parts.extend([actor_text] * 2)  # Medium weight for actors

        # Add directors
        if features.get('directors'):
            director_text = ' '.join(features['directors'])
            query_parts.append(director_text)

        # Add keywords
        if features.get('keywords'):
            keyword_text = ' '.join(features['keywords'])
            query_parts.append(keyword_text)

        # Add mood
        if features.get('mood'):
            mood_text = ' '.join(features['mood'])
            query_parts.append(mood_text)

        query_text = ' '.join(query_parts)

        if not query_text.strip():
            # Fallback to original query if no features extracted
            query_text = features.get('original_query', '')

        # Transform query using the fitted vectorizer
        query_vector = self.vectorizer.transform([query_text])

        return query_vector

    def search(self, features: Dict, k: int = 10) -> List[Dict]:
        """
        Perform keyword search using TF-IDF.

        Args:
            features: Dictionary of extracted features
            k: Number of results to return

        Returns:
            List of search results with scores
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Create query vector
        query_vector = self._create_query_vector(features)

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include results with some similarity
                result = {
                    'index': int(idx),
                    'tfidf_score': float(similarities[idx]),
                    'title': self.metadata_df.iloc[idx]['title'],
                    'type': self.metadata_df.iloc[idx].get('type', ''),
                    'listed_in': self.metadata_df.iloc[idx].get('listed_in', ''),
                    'cast': self.metadata_df.iloc[idx].get('cast', ''),
                    'description': self.metadata_df.iloc[idx].get('description', ''),
                    'platform': self.metadata_df.iloc[idx].get('platform', ''),
                    'release_year': self.metadata_df.iloc[idx].get('release_year', ''),
                    'show_id': self.metadata_df.iloc[idx].get('show_id', '')
                }
                results.append(result)

        return results

    def get_feature_importance(self, features: Dict) -> Dict[str, float]:
        """
        Calculate feature importance for the given query.

        Args:
            features: Dictionary of extracted features

        Returns:
            Dictionary of feature importance scores
        """
        importance = defaultdict(float)

        # Calculate importance based on feature presence and weights
        for feature_type, weight in self.feature_weights.items():
            feature_list = features.get(feature_type, [])
            if isinstance(feature_list, list) and feature_list:
                importance[feature_type] = weight * len(feature_list)
            elif isinstance(feature_list, str) and feature_list:
                importance[feature_type] = weight

        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}

        return dict(importance)

    def filter_by_content_type(self, results: List[Dict], content_type: str) -> List[Dict]:
        """
        Filter results by content type (movies, tv_shows, or both).

        Args:
            results: List of search results
            content_type: Content type filter ('movies', 'tv_shows', or 'both')

        Returns:
            Filtered results
        """
        if content_type == 'both':
            return results

        filtered_results = []
        for result in results:
            result_type = result.get('type', '').lower()
            if content_type == 'movies' and 'movie' in result_type:
                filtered_results.append(result)
            elif content_type == 'tv_shows' and ('tv' in result_type or 'show' in result_type):
                filtered_results.append(result)

        return filtered_results

    def get_statistics(self) -> Dict:
        """
        Get statistics about the TF-IDF index.

        Returns:
            Dictionary with index statistics
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return {}

        stats = {
            'total_documents': self.tfidf_matrix.shape[0],
            'vocabulary_size': self.tfidf_matrix.shape[1],
            'avg_document_length': self.tfidf_matrix.sum(axis=1).mean(),
            'sparsity': 1 - (self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))
        }

        return stats


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'title': ['The Dark Knight', 'Inception', 'Pulp Fiction', 'The Shawshank Redemption'],
        'cast': ['Christian Bale, Heath Ledger', 'Leonardo DiCaprio, Marion Cotillard', 'John Travolta, Samuel L. Jackson', 'Tim Robbins, Morgan Freeman'],
        'description': ['Batman fights Joker', 'Thief enters dreams', 'Gangster story', 'Prison escape story'],
        'listed_in': ['Action, Crime', 'Sci-Fi, Thriller', 'Crime, Drama', 'Drama'],
        'type': ['Movie', 'Movie', 'Movie', 'Movie'],
        'platform': ['Netflix', 'Netflix', 'Amazon Prime', 'Disney Plus']
    })

    # Initialize search engine
    search_engine = KeywordSearchEngine()

    # Build index
    search_engine.build_index(sample_data)

    # Test search
    test_features = {
        'genres': ['action', 'crime'],
        'actors': ['christian bale'],
        'keywords': [],
        'content_type': 'movies'
    }

    results = search_engine.search(test_features, k=3)

    print("Search Results:")
    for result in results:
        print(f"- {result['title']}: {result['tfidf_score']:.3f}")

    print(f"\nIndex Statistics: {search_engine.get_statistics()}")
