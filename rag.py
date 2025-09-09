import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

# -----------------------------
# Step 0: Configuration
# -----------------------------
platform_files = {
    "Amazon": "faiss_indexes/amazon_prime_index.faiss",
    "Netflix": "faiss_indexes/netflix_index.faiss",
    "Disney": "faiss_indexes/disney_plus_index.faiss"
}

user_history_files = {
    "Amazon": "user_history/Simulated_Amazon_User_Histories.csv",
    "Netflix": "user_history/Simulated_Netflix_User_Histories.csv",
    "Disney": "user_history/Simulated_Disney_User_Histories.csv"
}

full_metadata_file = "simulated_user_histories_all_platforms.csv"  # contains platform column
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
k = 5  # top-K recommendations

# -----------------------------
# Step 1: Terminal input
# -----------------------------
user_id = input("Enter user ID (e.g., user_1): ").strip()
platform = input("Select platform (Amazon, Netflix, Disney): ").strip()
query_text = input("Enter your query (e.g., action movies): ").strip()

if platform not in platform_files:
    raise ValueError("Invalid platform selected. Choose from Amazon, Netflix, Disney.")

# -----------------------------
# Step 2: Load data
# -----------------------------
user_history_df = pd.read_csv(user_history_files[platform])
full_metadata_df = pd.read_csv(full_metadata_file)

# Filter full metadata for selected platform
platform_metadata = full_metadata_df[full_metadata_df['platform'] == platform].reset_index(drop=True)
print(platform_metadata[0:5])
index = faiss.read_index(platform_files[platform])

print(f"FAISS index ntotal: {index.ntotal}")
print(f"Filtered metadata length: {len(platform_metadata)}")

# -----------------------------
# Step 3: Build user profile embedding
# -----------------------------
user_history = user_history_df[user_history_df['user_id'] == user_id]

if not user_history.empty:
    user_profile = np.mean(embed_model.encode(user_history['description'].tolist(), convert_to_numpy=True), axis=0)
else:
    user_profile = None  # no history

# -----------------------------
# Step 4: Encode query
# -----------------------------
query_embedding = embed_model.encode([query_text], convert_to_numpy=True)[0]
final_query_vector = query_embedding if user_profile is None else user_profile + query_embedding
final_query_vector = np.expand_dims(final_query_vector, axis=0).astype('float32')

# -----------------------------
# Step 5: Retrieve top-K items from FAISS
# -----------------------------
distances, indices = index.search(final_query_vector, k)

# Map indices to filtered metadata
valid_indices = [i for i in indices[0] if 0 <= i < len(platform_metadata)]
retrieved_items = platform_metadata.iloc[valid_indices]

print("\nTop Recommended Items:")
print(retrieved_items[['title', 'description', 'category']])

# -----------------------------
# Optional Step 6: Generate LLM rationale
# -----------------------------
# use_llm = input("\nDo you want an explanation for recommendations? (y/n): ").strip().lower()
# if use_llm == "y":
#     client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # replace with your key
#     prompt = f"""
#     User query: {query_text}
#     User profile: Aggregated interactions from user {user_id} (if any)
#     Recommended items:
#     {retrieved_items[['title', 'description', 'listed_in']].to_dict(orient='records')}

#     Write a clear, friendly explanation for why these items are recommended.
#     """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#     )
#     rationale = response.choices[0].message.content
#     print("\nRecommendation Rationale:\n", rationale)
