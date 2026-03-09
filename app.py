# app.py
import streamlit as st
import pandas as pd
import re, ast
from model_code import recommend, semantic_recommend, recommend_by_genre_from_tags, process_text, parse_genres
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"
    df = pd.read_csv(url)

    # Preprocess exactly like your Colab
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[['title','overview','genres','tagline','vote_average','popularity']]
    df = df.dropna(subset=['title'])
    df['overview'] = df['overview'].fillna('')
    df['tagline'] = df['tagline'].fillna('')

    # Convert genres field from list‑string to space‑separated text
    df['genres'] = df['genres'].apply(parse_genres)

    # Combine text into tags & clean it
    df['tags'] = df['overview'] + ' ' + df['genres'] + ' ' + df['tagline']
    df['tags'] = df['tags'].apply(process_text)

    # TF‑IDF vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])

    # SentenceTransformer embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['tags'].tolist(), show_progress_bar=True)

    # Build indices map
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    return df, tfidf_matrix, indices, embeddings

df, tfidf_matrix, indices, embeddings = load_data()

st.title("🎬 Movie Recommender App")

# ---------------- Movie‑based recommendations ---------------- #
st.subheader("Search by movie")
selected_movie = st.selectbox("Choose a movie:", df['title'].tolist())

if st.button("Recommend"):
    recs = semantic_recommend(selected_movie, df, indices, embeddings)
    st.write(f"**Movies similar to '{selected_movie}':**")
    for r in recs:
        st.text(r)

st.markdown("---")

# ---------------- Genre‑based recommendations ---------------- #
st.subheader("Search by genre")
genre_list = sorted(df['genres'].unique().tolist())
selected_genre = st.selectbox("Choose a genre:", genre_list)

if st.button("Recommend by Genre"):
    genre_recs = recommend_by_genre_from_tags(selected_genre, df)
    st.write(f"**Movies in '{selected_genre}' genre:**")
    for g in genre_recs:
        st.text(g)
