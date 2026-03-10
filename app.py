# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import ast
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

st.title("🎬 Movie Recommendation System")

# ------------------- Google Drive CSV ------------------- #
file_id = "1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"
url = f"https://drive.google.com/uc?id={file_id}"
csv_path = "movies_metadata.csv"
gdown.download(url, csv_path, quiet=False)

# ------------------- Load CSV ------------------- #
df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8", on_bad_lines="skip")

# ------------------- Clean & prepare ------------------- #
df = df.drop_duplicates().reset_index(drop=True)
df = df[['title','overview','genres','tagline','vote_average','popularity']]
df = df.dropna(subset=['title'])
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')

# ------------------- NLTK setup ------------------- #
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------- Genre parsing & text preprocessing ------------------- #
def parse_genres(x):
    try:
        if isinstance(x, list):
            return ' '.join([i['name'] for i in x])
        elif isinstance(x, str):
            data = ast.literal_eval(x)
            return ' '.join([i['name'] for i in data])
        else:
            return ''
    except:
        return ''

def process_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['genres'] = df['genres'].apply(parse_genres)
df['tags'] = (df['overview'] + ' ' + df['genres'] + ' ' + df['tagline']).apply(process_text)

# ------------------- Build indices & TF-IDF matrix ------------------- #
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ------------------- Recommendation functions ------------------- #
def recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

def recommend_by_genre_from_tags(user_genre, n=10):
    user_genre = user_genre.lower().strip()
    genre_filtered_df = df[df['tags'].str.contains(user_genre, case=False, na=False)]
    if genre_filtered_df.empty:
        return ['Genre not found']
    return genre_filtered_df['title'].head(n).values

# ------------------- Load semantic transformer & compute embeddings ------------------- #
@st.cache_resource(show_spinner=True)
def load_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_transformer_model()

@st.cache_data(show_spinner=True)
def compute_embeddings(tags):
    return model.encode(tags.tolist(), show_progress_bar=True)

embeddings = compute_embeddings(df['tags'])

def semantic_recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    movie_emb = embeddings[idx]
    sim_scores = np.dot(embeddings, movie_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(movie_emb))
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

# ------------------- Hybrid recommendation ------------------- #
def hybrid_recommend(title, n=10, alpha=0.5):
    """
    Combines TF-IDF and semantic embeddings.
    alpha: weight for TF-IDF (0-1), 1-alpha weight for semantic
    """
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    
    # TF-IDF similarity
    tfidf_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Semantic similarity
    movie_emb = embeddings[idx]
    sem_sim = np.dot(embeddings, movie_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(movie_emb))
    
    # Combine
    combined_sim = alpha * tfidf_sim + (1 - alpha) * sem_sim
    similar_idx = combined_sim.argsort()[::-1][1:n+1]
    
    return df['title'].iloc[similar_idx].values

# ------------------- Streamlit UI ------------------- #
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("Movie Based", "Semantic Movie Based", "Hybrid Recommendation", "Genre Based")
)

if option == "Movie Based":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    if st.button("Recommend"):
        results = recommend(movie_name)
        st.subheader("Recommended Movies (TF-IDF)")
        for movie in results:
            st.write(movie)

elif option == "Semantic Movie Based":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    if st.button("Recommend"):
        results = semantic_recommend(movie_name)
        st.subheader("Recommended Movies (Semantic Transformer)")
        for movie in results:
            st.write(movie)

elif option == "Hybrid Recommendation":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    alpha = st.slider("Weight for TF-IDF vs Semantic (higher = more TF-IDF)", 0.0, 1.0, 0.5)
    if st.button("Recommend"):
        results = hybrid_recommend(movie_name, alpha=alpha)
        st.subheader(f"Recommended Movies (Hybrid, α={alpha})")
        for movie in results:
            st.write(movie)

elif option == "Genre Based":
    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")
    if st.button("Recommend"):
        results = recommend_by_genre_from_tags(genre)
        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)
