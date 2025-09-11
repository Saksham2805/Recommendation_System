import os
import json
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv

class LLMFeatureExtractor:
    """
    Extracts features from user queries using LLM for enhanced recommendation system.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM feature extractor.

        Args:
            api_key: Google AI API key. If None, loads from environment variable.
        """
        load_dotenv()

        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable or pass it directly.")

        genai.configure(api_key=api_key)

        # Initialize the model - try different model names
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception:
            try:
                self.model = genai.GenerativeModel('gemini-1.5-pro')
            except Exception:
                try:
                    self.model = genai.GenerativeModel('gemini-pro')
                except Exception:
                    print("Warning: Could not initialize any Gemini model. LLM features will not work.")
                    self.model = None

        # Define the prompt template for feature extraction
        self.extraction_prompt = """
You are a movie and TV show recommendation assistant. Your task is to analyze user queries and extract key features that can be used for searching and recommending content.

For the given user query, extract the following features:
1. Genres: Any movie/TV genres mentioned (e.g., action, comedy, drama, horror, sci-fi, romance, thriller, etc.)
2. Actors/Cast: Any specific actors, actresses, or cast members mentioned
3. Directors: Any directors mentioned
4. Keywords: Other important keywords that describe the content (e.g., themes, settings, time periods, etc.)
5. Content Type: Whether they want movies, TV shows, or both
6. Mood/Tone: Any mood or tone preferences (e.g., funny, scary, romantic, intense, light-hearted, etc.)
7. Era/Year: Any time period or year preferences
8. Platform: Any specific streaming platform preferences

IMPORTANT: You must respond ONLY with a valid JSON object in this exact format:
{{
    "genres": ["genre1", "genre2"],
    "actors": ["actor1", "actor2"],
    "directors": ["director1"],
    "keywords": ["keyword1", "keyword2"],
    "content_type": "movies" or "tv_shows" or "both",
    "mood": ["mood1", "mood2"],
    "era": "specific era or year if mentioned",
    "platform": "platform name if mentioned"
}}

Do not include any other text, explanations, or formatting. Just the JSON object.

User Query: {query}

Extracted Features (JSON only):
"""

    def extract_features(self, user_query: str) -> Dict:
        """
        Extract features from the user query using LLM.

        Args:
            user_query: The user's search query

        Returns:
            Dictionary containing extracted features
        """
        try:
            # Format the prompt with the user query
            prompt = self.extraction_prompt.format(query=user_query)

            # Generate response from LLM
            response = self.model.generate_content(prompt)

            # Extract the JSON from the response
            response_text = response.text.strip()

            # Try to find JSON in the response (LLM might add extra text or code blocks)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]

                # Clean up the JSON string
                json_str = json_str.replace('\n', '').replace('\r', '').strip()

                try:
                    features = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Try multiple fallback strategies
                    json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                    json_str = json_str.replace('""', '"')  # Fix double double quotes
                    json_str = json_str.replace(',}', '}')  # Remove trailing commas

                    try:
                        features = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Fallback if JSON parsing still fails
                        features = self._fallback_extraction(user_query)
            else:
                # Fallback if JSON not found
                features = self._fallback_extraction(user_query)

            # Validate and clean the extracted features
            features = self._validate_features(features)

            return features

        except Exception as e:
            print(f"Error extracting features with LLM: {e}")
            # Fallback to simple keyword extraction
            return self._fallback_extraction(user_query)

    def _validate_features(self, features: Dict) -> Dict:
        """
        Validate and clean the extracted features.

        Args:
            features: Raw extracted features

        Returns:
            Validated and cleaned features
        """
        validated = {
            'genres': features.get('genres', []),
            'actors': features.get('actors', []),
            'directors': features.get('directors', []),
            'keywords': features.get('keywords', []),
            'content_type': features.get('content_type', 'both'),
            'mood': features.get('mood', []),
            'era': features.get('era', ''),
            'platform': features.get('platform', '')
        }

        # Ensure lists are actually lists and strings are strings
        for key in ['genres', 'actors', 'directors', 'keywords', 'mood']:
            if not isinstance(validated[key], list):
                validated[key] = [validated[key]] if validated[key] else []

        # Clean and lowercase all text
        for key in validated:
            if isinstance(validated[key], list):
                validated[key] = [item.lower().strip() for item in validated[key] if item]
            elif isinstance(validated[key], str):
                validated[key] = validated[key].lower().strip()

        return validated

    def _fallback_extraction(self, user_query: str) -> Dict:
        """
        Simple fallback feature extraction using basic keyword matching.

        Args:
            user_query: The user's search query

        Returns:
            Dictionary with basic extracted features
        """
        query_lower = user_query.lower()

        # Basic genre keywords
        genres = []
        genre_keywords = {
            'action': ['action', 'adventure'],
            'comedy': ['comedy', 'funny', 'humorous'],
            'drama': ['drama', 'dramatic'],
            'horror': ['horror', 'scary', 'thriller'],
            'sci-fi': ['sci-fi', 'science fiction', 'sci fi'],
            'romance': ['romance', 'romantic'],
            'documentary': ['documentary', 'doc'],
            'animation': ['animation', 'animated', 'cartoon']
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
            'platform': ''
        }

    def get_feature_weights(self, features: Dict) -> Dict[str, float]:
        """
        Assign weights to different features based on their importance.

        Args:
            features: Extracted features dictionary

        Returns:
            Dictionary mapping feature types to their weights
        """
        weights = {
            'genres': 0.4,      # Highest weight - genres are very important
            'actors': 0.25,     # High weight - specific actors are important
            'directors': 0.2,   # Medium-high weight - directors influence style
            'keywords': 0.1,    # Lower weight - general keywords
            'mood': 0.05        # Lowest weight - mood is subjective
        }

        # Adjust weights based on what's present
        present_features = [k for k, v in features.items() if v and (isinstance(v, list) and len(v) > 0 or isinstance(v, str) and v)]

        if len(present_features) > 0:
            # Normalize weights so they sum to 1
            total_weight = sum(weights[feat] for feat in present_features if feat in weights)
            if total_weight > 0:
                weights = {k: v/total_weight if k in present_features else 0 for k, v in weights.items()}

        return weights


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = LLMFeatureExtractor()

    # Test queries
    test_queries = [
        "I want to watch action movies with Tom Cruise",
        "Show me romantic comedies from the 90s",
        "Horror movies that are really scary",
        "Documentaries about space exploration"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        features = extractor.extract_features(query)
        weights = extractor.get_feature_weights(features)
        print(f"Features: {features}")
        print(f"Weights: {weights}")
