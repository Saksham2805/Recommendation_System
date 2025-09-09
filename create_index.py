import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os

# --------------------------
# STEP 1: Load the merged dataset
# --------------------------
merged_df = pd.read_csv("simulated_user_histories_all_platforms.csv")  # change to .csv if needed

# --------------------------
# STEP 2: Initialize embedding model
# --------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# --------------------------
# STEP 3: Build FAISS index per platform
# --------------------------
platform_indexes = {}
platform_metadata = {}

for platform in merged_df['platform'].unique():
    df_platform = merged_df[merged_df['platform'] == platform].reset_index(drop=True)
    texts = df_platform['title'].astype(str).tolist()
    
    # Optional: combine title + description for richer embeddings
    texts = (df_platform['title'] + " " + df_platform['description']).astype(str).tolist()
    
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    platform_indexes[platform] = index
    platform_metadata[platform] = df_platform.to_dict(orient='records')

print("✅ Multi-platform FAISS indexes are ready")

os.makedirs("faiss_indexes", exist_ok=True)

for platform, index in platform_indexes.items():
    faiss.write_index(index, f"faiss_indexes/{platform}_index.faiss")