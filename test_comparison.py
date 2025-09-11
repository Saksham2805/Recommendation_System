#!/usr/bin/env python3
"""
Test script to compare the original recommender (v1) with the new hybrid recommender.
"""

import pandas as pd
import os
import sys
from datetime import datetime

def test_recommenders():
    """Test both versions of the recommender system."""

    print("🎬 RECOMMENDER SYSTEM COMPARISON TEST")
    print("="*60)

    # Test parameters
    test_user_id = "user_01"
    test_query = "action movies with Tom Cruise"
    test_platform = None  # All platforms
    test_k = 3

    # Check if user history exists
    history_file = f"user_histories/{test_user_id}_history.csv"
    if not os.path.exists(history_file):
        print(f"❌ User history file not found: {history_file}")
        print("Please ensure user history files exist in the user_histories/ directory.")
        return

    print(f"📋 Test Parameters:")
    print(f"   User ID: {test_user_id}")
    print(f"   Query: '{test_query}'")
    print(f"   Platform: All")
    print(f"   Recommendations: {test_k}")
    print()

    # Test Original Version (v1)
    print("🔄 TESTING ORIGINAL VERSION (retrieve_elements_v1.py)")
    print("-" * 50)

    try:
        from retrieve_elements_v1 import PersonalizedRecommender as OriginalRecommender

        original_recommender = OriginalRecommender()
        original_history = original_recommender.load_user_history(test_user_id)

        if original_history is not None:
            original_preferences = original_recommender.extract_user_preferences(original_history)
            original_enhanced_query = original_recommender.enhance_query(test_query, original_preferences)

            print("Searching with original system...")
            original_results = original_recommender.get_recommendations(
                original_enhanced_query, test_platform, test_k, original_history
            )

            print(f"✅ Original system found results for {len(original_results)} platforms")

            # Display original recommendations
            original_recommender.display_recommendations(original_results, original_preferences, test_query)
            print()
        else:
            print("❌ Failed to load user history for original system")
            original_results = {}

    except Exception as e:
        print(f"❌ Error testing original system: {e}")
        original_results = {}

    print()

    # Test New Hybrid Version
    print("🆕 TESTING NEW HYBRID VERSION (retrieve_elements.py)")
    print("-" * 50)

    try:
        from retrieve_elements import HybridRecommender

        hybrid_recommender = HybridRecommender()
        hybrid_history = hybrid_recommender.load_user_history(test_user_id)

        if hybrid_history is not None:
            print("Searching with hybrid system...")
            hybrid_results = hybrid_recommender.get_recommendations(
                test_query, test_platform, test_k, hybrid_history
            )

            print(f"✅ Hybrid system found results for {len(hybrid_results.get('results', {}))} platforms")

            # Display hybrid recommendations
            hybrid_recommender.display_recommendations(hybrid_results, test_query)

            # Display search statistics
            if hybrid_results.get('search_stats'):
                stats = hybrid_results['search_stats']
                print("📊 Hybrid Search Statistics:")
                print(f"   🔤 Keyword results: {stats['keyword_results']}")
                print(f"   🧠 Semantic results: {stats['semantic_results']}")
                print(f"   ⚖️ Merged results: {stats['merged_results']}")
                print(f"   🏆 Final recommendations: {stats['final_results']}")

        else:
            print("❌ Failed to load user history for hybrid system")
            hybrid_results = {}

    except Exception as e:
        print(f"❌ Error testing hybrid system: {e}")
        hybrid_results = {}

    print()

    # Comparison Summary
    print("📊 COMPARISON SUMMARY")
    print("-" * 50)

    original_total = sum(len(platform_results) for platform_results in original_results.values()) if original_results else 0
    hybrid_total = sum(len(platform_results) for platform_results in hybrid_results.get('results', {}).values()) if hybrid_results else 0

    print(f"Original System Results: {original_total}")
    print(f"Hybrid System Results: {hybrid_total}")

    if hybrid_results and hybrid_results.get('features'):
        features = hybrid_results['features']
        print("🎭 Extracted Features:")
        if features.get('genres'):
            print(f"   Genres: {', '.join(features['genres'])}")
        if features.get('actors'):
            print(f"   Actors: {', '.join(features['actors'])}")
        if features.get('content_type') != 'both':
            print(f"   Content Type: {features['content_type']}")

    print()

    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 50)
    print("1. Compare the quality and relevance of recommendations between both systems")
    print("2. Note how the hybrid system extracts specific features from the query")
    print("3. Observe the search statistics to understand the hybrid approach")
    print("4. Test with different queries to see feature extraction in action")
    print("5. Try queries with specific actors, genres, or content types")

    print("\n" + "="*60)
    print("✅ Comparison test completed!")

def run_individual_test():
    """Run a single test with user input."""

    print("🎬 INDIVIDUAL RECOMMENDER TEST")
    print("="*50)

    try:
        # Choose version
        print("Choose recommender version:")
        print("1. Original Version (v1)")
        print("2. New Hybrid Version")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            from retrieve_elements_v1 import PersonalizedRecommender as Recommender
            version_name = "Original"
        elif choice == "2":
            from retrieve_elements import HybridRecommender as Recommender
            version_name = "Hybrid"
        else:
            print("Invalid choice!")
            return

        print(f"\n🟢 Testing {version_name} Version")
        print("-" * 30)

        # Get user input
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

        # Initialize and run
        if choice == "1":
            recommender = Recommender()
            user_history = recommender.load_user_history(user_id)

            if user_history is None:
                print("Cannot proceed without user history.")
                return

            preferences = recommender.extract_user_preferences(user_history)
            enhanced_query = recommender.enhance_query(user_query, preferences)

            print("Searching for recommendations...")
            results = recommender.get_recommendations(enhanced_query, platform, k, user_history)
            recommender.display_recommendations(results, preferences, user_query)

        else:  # Hybrid version
            recommender = Recommender()
            user_history = recommender.load_user_history(user_id)

            if user_history is None:
                print("Cannot proceed without user history.")
                return

            print("Searching for recommendations...")
            results = recommender.get_recommendations(user_query, platform, k, user_history)
            recommender.display_recommendations(results, user_query)

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        test_recommenders()
    else:
        run_individual_test()
