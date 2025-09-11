# Hybrid Movie/TV Recommender System

An advanced recommendation system that combines multiple search techniques for better movie and TV show recommendations.

## 🚀 Features Implemented

### 1. LLM-Powered Feature Extraction
- Uses Google Gemini AI to extract features from user queries
- Identifies genres, actors, directors, content types, and more
- Falls back to basic keyword matching if LLM is unavailable

### 2. Hybrid Search Approach
- **Keyword Search**: TF-IDF based search for precise matching
- **Semantic Search**: Sentence transformer embeddings for contextual understanding
- **User History Integration**: Personalizes recommendations based on viewing history

### 3. Weighted Ranking System
- Combines keyword and semantic search results
- Applies feature-specific boosts (genre matches, actor matches, etc.)
- Returns top-k recommendations with confidence scores

### 4. Enhanced User Experience
- Detailed search statistics
- Feature extraction display
- Platform-specific recommendations
- Comparison with original system

## 📁 File Structure

```
├── retrieve_elements.py          # New hybrid recommender (main file)
├── retrieve_elements_v1.py       # Original semantic-only recommender
├── llm_feature_extractor.py     # LLM-powered feature extraction
├── keyword_search.py            # TF-IDF keyword search engine
├── test_comparison.py           # Comparison testing script
├── create_index.py              # Index creation utilities
├── index_search.py              # Basic search utilities
├── prepare_data.py              # Data preparation scripts
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Google AI API Key (Optional)
For LLM feature extraction, set your Google AI API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```
Or create a `.env` file with:
```
GOOGLE_API_KEY=your_api_key_here
```

### 3. Prepare Data
Ensure you have:
- FAISS indexes in `faiss_indexes/` directory
- User history files in `user_histories/` directory
- Cleaned data in `data/cleaned/` directory

## 🎯 Usage

### Run the Hybrid Recommender
```bash
python retrieve_elements.py
```

### Compare with Original Version
```bash
python test_comparison.py --compare
```

### Test Individual Versions
```bash
python test_comparison.py
```

## 🔍 How It Works

### 1. Feature Extraction
The system analyzes user queries to extract:
- **Genres**: action, comedy, drama, horror, etc.
- **Actors**: specific cast members mentioned
- **Directors**: film directors
- **Content Type**: movies, TV shows, or both
- **Mood/Tone**: scary, funny, romantic, etc.
- **Era/Year**: time periods or specific years

### 2. Dual Search Strategy
- **Keyword Search**: Uses TF-IDF to find exact matches for extracted features
- **Semantic Search**: Uses sentence embeddings to find contextually similar content

### 3. Intelligent Ranking
- Combines results from both search methods
- Applies weights: 40% keyword, 60% semantic (configurable)
- Adds boosts for feature matches (genre: +20%, actor: +15%, etc.)
- Filters out already-watched content

### 4. Personalization
- Analyzes user viewing history
- Extracts preferences (favorite genres, content types, etc.)
- Enhances queries with personal context

## 📊 Search Statistics

The hybrid system provides detailed statistics:
- 🔤 Keyword results found
- 🧠 Semantic results found
- ⚖️ Total merged results
- 🏆 Final recommendations

## 🔄 Comparison with Original System

| Feature | Original (v1) | Hybrid (v2) |
|---------|---------------|-------------|
| Search Method | Semantic only | Hybrid (Keyword + Semantic) |
| Feature Extraction | Basic | LLM-powered |
| Personalization | Query enhancement | Full history analysis |
| Ranking | Single score | Weighted multi-factor |
| Statistics | Basic | Detailed |

## 🎭 Example Queries

Try these queries to see the system in action:

1. **"action movies with Tom Cruise"**
   - Extracts: genre="action", actor="Tom Cruise", content_type="movies"

2. **"romantic comedies from the 90s"**
   - Extracts: genres=["romance", "comedy"], era="90s"

3. **"scary horror movies"**
   - Extracts: genres=["horror"], mood=["scary"]

4. **"documentaries about space"**
   - Extracts: genre="documentary", keywords=["space"]

## ⚙️ Configuration

### Search Weights
Modify weights in `retrieve_elements.py`:
```python
self.search_weights = {
    'keyword': 0.4,    # Weight for keyword search
    'semantic': 0.6    # Weight for semantic search
}
```

### Feature Weights
Adjust feature importance in `keyword_search.py`:
```python
self.feature_weights = {
    'genres': 0.4,      # Highest weight
    'actors': 0.25,     # High weight
    'directors': 0.2,   # Medium weight
    'keywords': 0.1,    # Lower weight
    'mood': 0.05        # Lowest weight
}
```

## 🚀 Future Enhancements

1. **BM25 Integration**: Replace TF-IDF with BM25 for better term frequency handling
2. **Advanced LLM Features**: More sophisticated prompt engineering
3. **Collaborative Filtering**: User-user and item-item similarity
4. **Real-time Learning**: Update user preferences based on interactions
5. **Multi-modal Search**: Include images, trailers, and reviews

## 📈 Performance Notes

- **Index Building**: TF-IDF index is built once at startup
- **Search Speed**: Hybrid search is fast due to pre-computed embeddings
- **Memory Usage**: Optimized for large datasets with sparse matrices
- **Fallback Support**: Works without LLM API key using basic extraction

## 🤝 Contributing

1. Test your changes with `test_comparison.py`
2. Ensure backward compatibility with original system
3. Add documentation for new features
4. Update this README with any new functionality

## 📄 License

This project is part of the Recommendation System repository. See the main repository for licensing information.
