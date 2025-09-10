#!/usr/bin/env python3
"""
Test script for retrieve_elements.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieve_elements import PersonalizedRecommender

def test_recommender():
    """Test the personalized recommender functionality."""
    print("Testing Personalized Recommender...")

    # Initialize recommender
    recommender = PersonalizedRecommender()

    # Test user history loading
    user_history = recommender.load_user_history("user_01")
    if user_history is None:
        print("❌ Failed to load user history")
        return False

    print(f"✅ Loaded user history with {len(user_history)} entries")

    # Test preference extraction
    preferences = recommender.extract_user_preferences(user_history)
    print(f"✅ Extracted preferences: {preferences}")

    # Test query enhancement
    test_query = "action movies"
    enhanced_query = recommender.enhance_query(test_query, preferences)
    print(f"✅ Enhanced query: {enhanced_query}")

    # Test recommendations
    results = recommender.get_recommendations(enhanced_query, platform="Netflix", k=3, user_history=user_history)
    print(f"✅ Got recommendations: {len(results)} platforms")

    # Display results
    recommender.display_recommendations(results, preferences, test_query)

    return True

if __name__ == "__main__":
    success = test_recommender()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
