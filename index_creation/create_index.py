import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import Tuple
import re

class StreamingPlatformIndexer:
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
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and clean
        text = str(text)
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?]', ' ', text)
        return text.strip()
    
    def create_combined_text(self, row: pd.Series) -> str:
        """
        Combine multiple text fields into a single searchable text.
        
        Args:
            row: A pandas Series representing a single row from the CSV
            
        Returns:
            Combined text string for embedding generation
        """
        # Combine relevant text fields with appropriate weights
        title = self.clean_text(row.get('title', ''))
        cast_info = self.clean_text(row.get('cast', ''))
        description = self.clean_text(row.get('description', ''))
        listed_in = self.clean_text(row.get('listed_in', ''))
        show_type = self.clean_text(row.get('type', ''))
        duration = self.clean_text(row.get('duration', ''))
        
        # Create a weighted combination - title and description are most important
        combined_parts = []
        
        # Add title multiple times for higher weight
        if title:
            combined_parts.extend([title] * 3)
        
        # Add description with high weight
        if description:
            combined_parts.extend([description] * 2)
        
        # Add other fields once
        if cast_info:
            combined_parts.append(f"Cast: {cast_info}")
        if listed_in:
            combined_parts.append(f"Genre: {listed_in}")
        if show_type:
            combined_parts.append(f"Type: {show_type}")
        if duration:
            combined_parts.append(f"Duration: {duration}")
            
        return " ".join(combined_parts)
    
    def load_and_process_csv(self, csv_path: str, platform_name: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load CSV file and create embeddings.
        
        Args:
            csv_path: Path to the CSV file
            platform_name: Name of the streaming platform
            
        Returns:
            Tuple of (embeddings array, processed dataframe)
        """
        print(f"Loading and processing {platform_name} data from {csv_path}")
        
        # Load CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from {platform_name}")
        
        # Add platform information
        df['platform'] = platform_name
        
        # Create combined text for each row
        print("Creating combined text for embedding...")
        df['combined_text'] = df.apply(self.create_combined_text, axis=1)
        
        # Filter out rows with empty combined text
        df = df[df['combined_text'].str.len() > 0].reset_index(drop=True)
        print(f"Processing {len(df)} records after filtering")
        
        # Generate embeddings
        print("Generating embeddings...")
        combined_texts = df['combined_text'].tolist()
        
        # Process in batches to avoid memory issues
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(combined_texts), batch_size):
            batch = combined_texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(combined_texts)-1)//batch_size + 1}")
        
        embeddings = np.vstack(embeddings)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings, df
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = 'flat') -> faiss.Index:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            
        Returns:
            FAISS index
        """
        print(f"Creating FAISS index of type: {index_type}")
        
        if index_type == 'flat':
            # Simple flat index - exact search
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
            
        elif index_type == 'ivf':
            # IVF index - approximate search, faster for large datasets
            nlist = min(100, max(1, embeddings.shape[0] // 100))  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train the index
            print("Training IVF index...")
            index.train(embeddings)
            
        elif index_type == 'hnsw':
            # HNSW index - hierarchical navigable small world
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 40
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        print("Adding embeddings to index...")
        index.add(embeddings)
        
        print(f"Index created with {index.ntotal} vectors")
        return index
    
    def process_platform(self, csv_path: str, platform_name: str, index_type: str = 'flat'):
        """
        Complete pipeline to process a platform's data and create index.
        
        Args:
            csv_path: Path to CSV file
            platform_name: Name of the platform
            index_type: Type of FAISS index to create
        """
        # Load and process data
        embeddings, df = self.load_and_process_csv(csv_path, platform_name)
        
        # Create FAISS index
        index = self.create_faiss_index(embeddings, index_type)
        
        # Store index and metadata
        self.indexes[platform_name] = index
        self.metadata[platform_name] = df
        
        print(f"✅ Successfully created index for {platform_name}")
        print("-" * 50)
    
    def save_indexes(self, output_dir: str = 'faiss_indexes'):
        """
        Save FAISS indexes and metadata to disk.
        
        Args:
            output_dir: Directory to save the indexes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for platform_name in self.indexes.keys():
            # Save FAISS index
            index_path = os.path.join(output_dir, f"{platform_name}_index.faiss")
            faiss.write_index(self.indexes[platform_name], index_path)
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"{platform_name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata[platform_name], f)
            
            print(f"Saved {platform_name} index and metadata")
    
    def get_index_stats(self):
        """Get statistics about the created indexes."""
        stats = {}
        for platform, index in self.indexes.items():
            stats[platform] = {
                'total_vectors': index.ntotal,
                'embedding_dimension': self.embedding_dim,
                'index_type': type(index).__name__,
                'metadata_records': len(self.metadata[platform]) if platform in self.metadata else 0
            }
        return stats

# Example usage
def main():
    # Initialize the indexer
    indexer = StreamingPlatformIndexer()

    # Process each platform (update these paths to your actual CSV files)
    platforms = {
        'netflix': 'data/cleaned/Netflix_data.csv',
        'disney': 'data/cleaned/Disney_Plus_data.csv',
        'amazon_prime': 'data/cleaned/Amazon_Prime_data.csv'
    }

    # Create indexes for each platform
    for platform_name, csv_path in platforms.items():
        try:
            indexer.process_platform(csv_path, platform_name, index_type='flat')
        except FileNotFoundError:
            print(f"❌ File not found: {csv_path}")
        except Exception as e:
            print(f"❌ Error processing {platform_name}: {str(e)}")
    
    # Save indexes
    indexer.save_indexes()
    
    # Display statistics
    print("\n📊 Index Statistics:")
    stats = indexer.get_index_stats()
    for platform, stat in stats.items():
        print(f"{platform.upper()}:")
        print(f"  - Total vectors: {stat['total_vectors']}")
        print(f"  - Embedding dimension: {stat['embedding_dimension']}")
        print(f"  - Index type: {stat['index_type']}")
        print(f"  - Metadata records: {stat['metadata_records']}")

if __name__ == "__main__":
    main()
