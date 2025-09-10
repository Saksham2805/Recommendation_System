import faiss
import pickle
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import Dict

class StreamingSearchEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the indexer with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.indexes = {}
        self.metadata = {}

    def load_indexes(self, output_dir: str = 'faiss_indexes'):
        """
        Load FAISS indexes and metadata from disk.
        
        Args:
            output_dir: Directory containing the saved indexes
        """
        for file in os.listdir(output_dir):
            if file.endswith('_index.faiss'):
                platform_name = file.replace('_index.faiss', '')
                
                # Load FAISS index
                index_path = os.path.join(output_dir, file)
                self.indexes[platform_name] = faiss.read_index(index_path)
                
                # Load metadata
                metadata_path = os.path.join(output_dir, f"{platform_name}_metadata.pkl")
                with open(metadata_path, 'rb') as f:
                    self.metadata[platform_name] = pickle.load(f)
                
                print(f"Loaded {platform_name} index and metadata")
    
    def search(self, query: str, platform: str = None, k: int = 5) -> Dict:
        """
        Search for similar content across platforms.
        
        Args:
            query: Search query
            platform: Specific platform to search (None for all platforms)
            k: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        results = {}
        
        platforms_to_search = [platform] if platform else list(self.indexes.keys())
        
        for plt in platforms_to_search:
            if plt not in self.indexes:
                print(f"Warning: No index found for platform {plt}")
                continue
                
            # Search in FAISS index
            scores, indices = self.indexes[plt].search(query_embedding, k)
            
            # Get metadata for results
            platform_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid result
                    result_row = self.metadata[plt].iloc[idx]
                    platform_results.append({
                        'score': float(score),
                        'show_id': result_row.get('show_id', ''),
                        'title': result_row.get('title', ''),
                        'type': result_row.get('type', ''),
                        'cast': result_row.get('cast', ''),
                        'duration': result_row.get('duration', ''),
                        'listed_in': result_row.get('listed_in', ''),
                        'description': result_row.get('description', ''),
                        'platform': plt
                    })
            
            results[plt] = platform_results
        
        return results
def main():

    # Initialize the indexer
    engine = StreamingSearchEngine()

    # Load existing indexes if available
    engine.load_indexes()

    # Example search
    if engine.indexes:
        print("\n🔍 Example Search:")
        query = "drama movies with comedy"
        results = engine.search(query, k=3)
        
        for platform, platform_results in results.items():
            print(f"\n{platform.upper()} Results:")
            for i, result in enumerate(platform_results, 1):
                print(f"{i}. {result['title']} (Score: {result['score']:.3f})")
                print(f"   Type: {result['type']}, Duration: {result['duration']}")
                print(f"   Description: {result['description'][:100]}...")
if __name__ == "__main__":
    main()