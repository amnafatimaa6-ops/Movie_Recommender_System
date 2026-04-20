# 🎬 MOVIE RECOMMENDER SYSTEM

> Scroll down to check the demos to see how each method works in practice!

A Streamlit-based Movie Recommendation System that suggests movies using multiple recommendation techniques including TF-IDF similarity, Semantic embeddings (Sentence Transformers), Hybrid recommendations, and Genre-based filtering.


The application allows users to select a movie or genre and receive similar movie suggestions along with ratings, similarity scores, popularity, and overview.

----------------------------------------------------------------

🚀 Features

TF-IDF Based Recommendation

Uses TF-IDF vectorization on movie metadata.

Calculates similarity using cosine similarity.

------------------------------------------------------------------

Semantic Recommendation

Uses Sentence Transformers (all-MiniLM-L6-v2) to generate embeddings.

Captures deeper semantic meaning of movie descriptions.

------------------------------------------------------------------
 
Hybrid Recommendation
Combines TF-IDF similarity and semantic similarity.
Provides more accurate and balanced recommendations.

------------------------------------------------------------------

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

## 🎥 Demo

Check out the demos of the Movie Recommendation System:

### TF-IDF Based Recommendation
![TF-IDF Demo](TF-IDF%20demo.gif)

---

### Semantic Transformer Recommendation
![Semantic Demo](Semantic-demo.gif)

---

### Hybrid Recommendation
![Hybrid Demo](Hybrid-demo.gif)

---

### Another Hybrid Demo Example
![Hybrid Demo 2](Hybrid-demo2.gif)

---

### Genre-Based Recommendation
![Genre Demo](Genre-based_demo.gif)

---

### 🔹 Updated Demo 1
![Updated Demo 1](updateddemo1.gif)

---

### 🔹 Updated Demo 2
![Updated Demo 2](updateddemo2.gif)

> Watch these GIFs to see the system in action!

### 🆕 Update Note

> Added **YouTube trailer integration layer**, allowing real-time retrieval and embedding of official trailers within the recommendation workflow, improving interpretability and user engagement with recommended content.

## 🧠 Recommendation Methods Summary

- **TF-IDF Based Recommendation**  
  Uses word matching in movie metadata (overview, genres, tagline).  
  ⚠️ It mainly recommends movies with **similar words**, so it’s more surface-level.

- **Semantic Transformer Based Recommendation**  
  Uses **SentenceTransformer embeddings** to understand the **context and meaning** of movie descriptions.  
  ✅ Recommends movies that are **contextually similar**, even if they don’t share exact words.

- **Hybrid Recommendation**  
  Combines TF-IDF and semantic similarity to balance **word matching** and **context understanding**.  
  🎯 Provides more **accurate and meaningful movie suggestions**.

> live demo
  > https://movierecommendersystem-qwaynni8hafwvr4t463sdz.streamlit.app/

