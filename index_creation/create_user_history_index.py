import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer

class UserHistoryIndexer:
    """
    Creates FAISS index for user movie history with movie details.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the indexer with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model to use.
        """
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def combine_features(self, movie: Dict[str, Any]) -> str:
        """
        Combine movie features into a single text string for indexing.

        Args:
            movie: Dictionary containing movie data.

        Returns:
            Combined text string.
        """
        title = movie['title']
        genre = movie.get('genre', '')
        description = movie.get('description', '')
        count = movie.get('count', 1)

        # Repeat title based on count to give more weight to frequently watched movies
        repeated_title = ' '.join([title] * count)

        # Combine all features
        combined = f"{repeated_title} {genre} {description}".strip()
        return combined

    def create_index_from_json(self, json_file_path: str, index_name: str = None):
        """
        Create FAISS index from a JSON file containing movie data.

        Args:
            json_file_path: Path to the JSON file.
            index_name: Name for the index (used in saved files). If None, uses file basename.
        """
        print(f"Loading movie data from {json_file_path}...")

        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            movies = json.load(f)

        print(f"Loaded {len(movies)} movies.")

        # Prepare texts and metadata
        texts = []
        metadata = []

        for movie in movies:
            combined_text = self.combine_features(movie)
            texts.append(combined_text)
            metadata.append(movie)

        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        print(f"Embeddings shape: {embeddings.shape}")

        # Create FAISS index
        print("Creating FAISS index...")
        index = faiss.IndexFlatIP(self.dimension)  # Inner product with normalized vectors = cosine similarity
        index.add(embeddings.astype('float32'))

        # Set index name
        if index_name is None:
            index_name = os.path.splitext(os.path.basename(json_file_path))[0]

        # Save index and metadata
        index_dir = 'faiss_indexes/saksham'
        os.makedirs(index_dir, exist_ok=True)

        index_path = os.path.join(index_dir, f"{index_name}_index.faiss")
        metadata_path = os.path.join(index_dir, f"{index_name}_metadata.pkl")

        # Save FAISS index
        faiss.write_index(index, index_path)

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
        print(f"Indexed {len(movies)} movies successfully.")

    def create_combined_index(self, json_files: List[str], index_name: str):
        """
        Create a combined FAISS index from multiple JSON files.

        Args:
            json_files: List of paths to JSON files.
            index_name: Name for the combined index.
        """
        print(f"Creating combined index '{index_name}' from {len(json_files)} files...")

        all_texts = []
        all_metadata = []

        for json_file in json_files:
            print(f"Processing {json_file}...")

            with open(json_file, 'r', encoding='utf-8') as f:
                movies = json.load(f)

            for movie in movies:
                combined_text = self.combine_features(movie)
                all_texts.append(combined_text)
                all_metadata.append(movie)

        print(f"Total movies to index: {len(all_texts)}")

        print("Generating embeddings...")
        embeddings = self.model.encode(all_texts, show_progress_bar=True)

        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        print(f"Embeddings shape: {embeddings.shape}")

        # Create FAISS index
        print("Creating FAISS index...")
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings.astype('float32'))

        # Save combined index and metadata
        index_dir = 'faiss_indexes/saksham'
        os.makedirs(index_dir, exist_ok=True)

        index_path = os.path.join(index_dir, f"{index_name}_index.faiss")
        metadata_path = os.path.join(index_dir, f"{index_name}_metadata.pkl")

        faiss.write_index(index, index_path)

        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)

        print(f"Combined index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
        print(f"Indexed {len(all_texts)} movies successfully.")

def main():
    """
    Main function to create user history indexes.
    """
    indexer = UserHistoryIndexer()

    detailed_dir = os.path.join('histories', 'detailed')

    # Create individual indexes for each user
    for json_file in os.listdir(detailed_dir):
        if json_file.endswith('_enriched.json'):
            json_path = os.path.join(detailed_dir, json_file)
            index_name = json_file.replace('_enriched.json', '')
            indexer.create_index_from_json(json_path, index_name)

    # Create a combined index with all movies
    json_files = [
        os.path.join(detailed_dir, f)
        for f in os.listdir(detailed_dir)
        if f.endswith('_enriched.json')
    ]
    if json_files:
        indexer.create_combined_index(json_files, 'combined_history')

if __name__ == "__main__":
    main()
