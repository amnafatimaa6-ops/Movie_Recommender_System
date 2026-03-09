# app.py
import streamlit as st
import pandas as pd
import re, ast
from model_code import recommend, semantic_recommend, recommend_by_genre_from_tags, process_text, parse_genres
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

@st.cache_data
def load_data():
    # ---- Google Drive IDs ----
    clean_file_id = "1VUTcCivOodVHSfwxSdeKVeNFazdJaCq-"  # Clean CSV (for recommendation)
    raw_file_id = "1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"    # Original dataset (raw)

    clean_url = f"https://drive.google.com/uc?id={clean_file_id}"
    raw_url = f"https://drive.google.com/uc?id={raw_file_id}"

    # Load datasets
    df_clean = pd.read_csv(clean_url)
    df_raw = pd.read_csv(raw_url)

    # Preprocess the clean dataset for recommendations
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    df_clean = df_clean[['title','overview','genres','tagline','vote_average','popularity']]
    df_clean = df_clean.dropna(subset=['title'])
    df_clean['overview'] = df_clean['overview'].fillna('')
    df_clean['tagline'] = df_clean['tagline'].fillna('')
    df_clean['genres'] = df_clean['genres'].apply(parse_genres)
    df_clean['tags'] = df_clean['overview'] + ' ' + df_clean['genres'] + ' ' + df_clean['tagline']
    df_clean['tags'] = df_clean['tags'].apply(process_text)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_clean['tags'])

    # SentenceTransformer embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_clean['tags'].tolist(), show_progress_bar=True)

    # Build indices map
    indices = pd.Series(df_clean.index, index=df_clean['title']).drop_duplicates()

    return df_clean, df_raw, tfidf_matrix, indices, embeddings

# Load both datasets
df_clean, df_raw, tfidf_matrix, indices, embeddings = load_data()

st.title("🎬 Movie Recommender App")

# ---------------- Raw dataset preview ---------------- #
st.subheader("Raw Dataset Preview (for reference)")
st.dataframe(df_raw.head(10))

st.markdown("---")

# ---------------- Movie‑based recommendations ---------------- #
st.subheader("Search by movie")
selected_movie = st.selectbox("Choose a movie:", df_clean['title'].tolist())

if st.button("Recommend"):
    recs = semantic_recommend(selected_movie, df_clean, indices, embeddings)
    st.write(f"**Movies similar to '{selected_movie}':**")
    for r in recs:
        st.text(r)

st.markdown("---")

# ---------------- Genre‑based recommendations ---------------- #
st.subheader("Search by genre")
genre_list = sorted(df_clean['genres'].unique().tolist())
selected_genre = st.selectbox("Choose a genre:", genre_list)

if st.button("Recommend by Genre"):
    genre_recs = recommend_by_genre_from_tags(selected_genre, df_clean)
    st.write(f"**Movies in '{selected_genre}' genre:**")
    for g in genre_recs:
        st.text(g)
