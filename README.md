##🎬Movie Recommender System

A Streamlit-based Movie Recommendation System that suggests movies using multiple recommendation techniques including TF-IDF similarity, Semantic embeddings (Sentence Transformers), Hybrid recommendations, and Genre-based filtering.

The application allows users to select a movie or genre and receive similar movie suggestions along with ratings, similarity scores, popularity, and overview.
🚀 Features
TF-IDF Based Recommendation
Uses TF-IDF vectorization on movie metadata.
Calculates similarity using cosine similarity.
Semantic Recommendation
Uses Sentence Transformers (all-MiniLM-L6-v2) to generate embeddings.
Captures deeper semantic meaning of movie descriptions.

Hybrid Recommendation
Combines TF-IDF similarity and semantic similarity.
Provides more accurate and balanced recommendations.

Genre Based Recommendation
Allows users to search movies by genre keywords.

Interactive UI

Built with Streamlit

Clean layout with movie ratings, similarity scores, popularity, and descriptions.

🧠 Recommendation Methods
1. TF-IDF Similarity

Uses Term Frequency – Inverse Document Frequency to convert movie text features into vectors and compares them using cosine similarity.

2. Semantic Similarity

Uses SentenceTransformer embeddings to understand contextual meaning in movie descriptions.

3. Hybrid Model

Combines both methods:

Hybrid Score = (TF-IDF Weight × TF-IDF Similarity) + (Semantic Weight × Semantic Similarity)

4. Genre Filtering

Filters movies by genre extracted from dataset metadata.

📂 Dataset
The dataset is downloaded automatically from Google Drive when the application runs.
Dataset contains:
Title
Overview
Genres
Tagline

Vote Average
Popularity
