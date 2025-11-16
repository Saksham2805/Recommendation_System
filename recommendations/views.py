from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from recommender import HybridRecommender


_recommender_instance: HybridRecommender | None = None


def get_recommender() -> HybridRecommender:
    """Return a singleton HybridRecommender instance.

    For now this uses the default configuration and the existing indexes,
    including the pre-built user history index for the sample user (saksham).
    """
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = HybridRecommender()
    return _recommender_instance


@api_view(["POST"])
@permission_classes([AllowAny])  # For quick testing; can tighten later
def query_recommendations(request):
    """Simple API to query recommendations.

    Expected JSON body:
    {
      "query": "sci-fi movies with aliens",
      "platform": "all" | "Netflix" | "Amazon Prime" | "Disney Plus" (optional),
      "k": 10 (optional, default 5),
      "user_id": "saksham" (optional; for now defaults to saksham)
    }
    """

    data = request.data or {}
    user_query = data.get("query", "").strip()
    if not user_query:
        return Response({"detail": "'query' is required."}, status=status.HTTP_400_BAD_REQUEST)

    platform = data.get("platform")
    if isinstance(platform, str) and platform.lower() == "all":
        platform = None

    try:
        k = int(data.get("k", 5))
    except (TypeError, ValueError):
        k = 5

    # For now, use the existing sample user ID "saksham" used by the repo
    # so we can reuse the pre-built history index and JSON files.
    user_id = data.get("user_id") or "saksham"

    recommender = get_recommender()

    # Load user history via existing helper
    user_history_df = recommender.load_user_history(user_id)

    results = recommender.get_recommendations(
        user_query=user_query,
        platform=platform,
        k=k,
        user_history=user_history_df,
    )

    # Flatten results into a list for the frontend
    flat_results = []
    grouped = results.get("results", {})
    for plt, plt_results in grouped.items():
        for item in plt_results:
            flat_results.append(
                {
                    "title": item.get("title"),
                    "platform": item.get("platform", plt),
                    "type": item.get("type"),
                    "listed_in": item.get("listed_in"),
                    "description": item.get("description"),
                    "release_year": item.get("release_year"),
                    "final_score": item.get("final_score"),
                    "semantic_score": item.get("semantic_score"),
                    "tfidf_score": item.get("tfidf_score", 0.0),
                }
            )

    response_payload = {
        "results": flat_results,
        "features": results.get("features", {}),
        "preferences": results.get("preferences", {}),
        "search_stats": results.get("search_stats", {}),
    }

    return Response(response_payload, status=status.HTTP_200_OK)
