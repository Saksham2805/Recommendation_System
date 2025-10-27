#!/usr/bin/env python3
"""
Hybrid User Search Script
Integrates LLM feature extraction with user history and hybrid search.
"""

import os
import json
from typing import Dict, List, Optional
from retrieve_elements import HybridRecommender
from llm_feature_extractor import LLMFeatureExtractor
from keyword_search import KeywordSearchEngine
import pandas as pd
from datetime import datetime

class UserHistoryLoader:
    """
    Loads and processes user viewing history from JSON files.
    """

    def __init__(self, history_dir: str = 'histories/detailed'):
        """
        Initialize the user history loader.

        Args:
            history_dir: Directory containing user history JSON files
        """
        self.history_dir = history_dir

    def load_user_history(self, user_id: str) -> Optional[pd.DataFrame]:
        """
        Load user's viewing history from JSON files.

        Args:
            user_id: User identifier

        Returns:
            DataFrame with user's history or None if not found
        """
        # Look for history files matching the user ID
        history_files = []
        if os.path.exists(self.history_dir):
            for file in os.listdir(self.history_dir):
                if user_id.lower() in file.lower() and file.endswith('.json'):
                    history_files.append(os.path.join(self.history_dir, file))

        if not history_files:
            print(f"No history files found for user {user_id}")
            return None

        all_history = []
        for file_path in history_files:
            try:
                with open(file_path, 'r') as f:
                    platform_history = json.load(f)

                platform_name = 'Netflix' if 'netflix' in file_path.lower() else 'Amazon Prime'

                # Convert to DataFrame format compatible with recommender
                for item in platform_history:
                    history_entry = {
                        'title': item.get('title', ''),
                        'type': item.get('type', ''),
                        'listed_in': item.get('genres', ''),
                        'cast': item.get('cast', ''),
                        'description': item.get('description', ''),
                        'director': item.get('director', ''),
                        'user_rating': item.get('rating', 0),
                        'watch_date': item.get('date_watched', datetime.now().strftime('%Y-%m-%d')),
                        'platform': platform_name,
                        'show_id': item.get('show_id', ''),
                        'duration': item.get('duration', ''),
                        'release_year': item.get('release_year', '')
                    }
                    all_history.append(history_entry)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        if not all_history:
            print(f"No valid history data found for user {user_id}")
            return None

        df = pd.DataFrame(all_history)
        print(f"Loaded {len(df)} entries from {user_id}'s history across {len(history_files)} platforms")
        return df

def main():
    """
    Main function to run hybrid search with user history.
    """
    print("🎬 HYBRID USER SEARCH RECOMMENDER")
    print("=" * 50)

    # Initialize components
    try:
        # Load user history handler
        history_loader = UserHistoryLoader()
        print("✅ User history loader initialized")

        # Initialize hybrid recommender
        recommender = HybridRecommender()
        print("✅ Hybrid recommender initialized")

    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        return

    # Get user input
    try:
        user_id = input("Enter user ID (e.g., saksham): ").strip()
        if not user_id:
            print("User ID cannot be empty!")
            return

        query = input("Enter your search query: ").strip()
        if not query:
            print("Query cannot be empty!")
            return

        platform_input = input("Enter platform (Netflix/Amazon Prime/Disney Plus) or 'all': ").strip()
        platform = None if platform_input.lower() == 'all' else platform_input

        k_input = input("Number of recommendations (default 5): ").strip()
        k = int(k_input) if k_input.isdigit() else 5

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    # Load user history
    print(f"📚 Loading user history for {user_id}...")
    user_history = history_loader.load_user_history(user_id)

    if user_history is None:
        print("❌ Could not load user history. Proceeding without personalization.")

    # Perform hybrid search
    print("🔍 Performing hybrid search...")
    try:
        results = recommender.get_recommendations(query, platform, k, user_history)

        # Display results
        recommender.display_recommendations(results, query)

    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
