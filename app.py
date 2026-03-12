import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import ast, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sentence_transformers import SentenceTransformer
import os

# ------------------- Streamlit UI ------------------- #
st.title("🎬 Movie Recommendation System")

# ------------------- Download & Load CSV ------------------- #
file_id = "1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"
url = f"https://drive.google.com/uc?id={file_id}"
csv_path = "movies_metadata.csv"

if not os.path.exists(csv_path):
    gdown.download(url, csv_path, quiet=False)

df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8", on_bad_lines="skip")

# ------------------- Data Cleaning ------------------- #
df = df.drop_duplicates().reset_index(drop=True)
df = df[['title','overview','genres','tagline','vote_average','popularity','release_date']]
df = df.dropna(subset=['title'])
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

# ------------------- NLTK Setup ------------------- #
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def parse_genres(x):
    try:
        if isinstance(x, list):
            return ' '.join([i['name'] for i in x])
        elif isinstance(x, str):
            data = ast.literal_eval(x)
            return ' '.join([i['name'] for i in data])
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

# ------------------- Build Indices & TF-IDF ------------------- #
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ------------------- Load / Save Sentence Transformer ------------------- #
model_path = "sentence_model"
embeddings_path = "movie_embeddings.npy"

@st.cache_resource(show_spinner=True)
def load_transformer():
    if os.path.exists(model_path):
        return SentenceTransformer(model_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save(model_path)
    return model

@st.cache_data(show_spinner=True)
def get_embeddings(model):
    if os.path.exists(embeddings_path):
        return np.load(embeddings_path)
    embs = model.encode(df['tags'].tolist(), show_progress_bar=True)
    np.save(embeddings_path, embs)
    return embs

transformer_model = load_transformer()
embeddings = get_embeddings(transformer_model)

# ------------------- Hybrid Recommendation ------------------- #
def hybrid_recommend(title, n=10, weights=None):
    if weights is None:
        weights = {'tfidf':0.4, 'semantic':0.4, 'popularity':0.2, 'genre':0.2}
    if title not in indices:
        return ['Movie not found']
    
    idx = indices[title]

    # TF-IDF similarity
    tfidf_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Semantic similarity
    movie_emb = embeddings[idx].reshape(1, -1)
    semantic_sim = cosine_similarity(movie_emb, embeddings).flatten()
    
    # Genre similarity
    genre_vec = tfidf.transform([df['genres'].iloc[idx]])
    genre_sim = cosine_similarity(genre_vec, tfidf.transform(df['genres'])).flatten()
    
    # Popularity normalization
    popularity_norm = df['popularity'].values / df['popularity'].max()

    # Weighted sum
    combined = (weights['tfidf']*tfidf_sim +
                weights['semantic']*semantic_sim +
                weights['genre']*genre_sim +
                weights['popularity']*popularity_norm)

    top_idx = combined.argsort()[::-1][1:n+1]
    return df['title'].iloc[top_idx].values

# ------------------- TF-IDF only ------------------- #
def recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

# ------------------- Semantic only ------------------- #
def semantic_recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    movie_emb = embeddings[idx].reshape(1, -1)
    sim = cosine_similarity(movie_emb, embeddings).flatten()
    top_idx = sim.argsort()[::-1][1:n+1]
    return df['title'].iloc[top_idx].values

# ------------------- Genre Based ------------------- #
def recommend_by_genre_from_tags(user_genre, n=10):
    user_genre = user_genre.lower().strip()
    genre_filtered_df = df[df['tags'].str.contains(user_genre, case=False, na=False)]
    if genre_filtered_df.empty:
        return ['Genre not found']
    return genre_filtered_df['title'].head(n).values

# ------------------- Streamlit Sidebar ------------------- #
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("TF-IDF Movie Based", "Semantic Movie Based", "Hybrid Movie Based", "Genre Based", "Semantic Search")
)

# Filters
st.sidebar.subheader("Optional Filters")
min_rating = st.sidebar.slider("Minimum rating", 0.0, 10.0, 0.0)
year_range = st.sidebar.slider("Release year range", int(df['release_year'].min()), int(df['release_year'].max()), (1980, 2024))

filtered_df = df[(df['vote_average'] >= min_rating) & 
                 (df['release_year'] >= year_range[0]) &
                 (df['release_year'] <= year_range[1])]

filtered_indices = pd.Series(filtered_df.index, index=filtered_df['title']).drop_duplicates()

# ------------------- UI Logic ------------------- #
if option == "TF-IDF Movie Based":
    movie_name = st.selectbox("Select a movie", filtered_df['title'].sort_values())
    if st.button("Recommend"):
        results = recommend(movie_name)
        st.subheader("Recommended Movies (TF-IDF)")
        for movie in results:
            st.write(movie)

elif option == "Semantic Movie Based":
    movie_name = st.selectbox("Select a movie", filtered_df['title'].sort_values())
    if st.button("Recommend"):
        results = semantic_recommend(movie_name)
        st.subheader("Recommended Movies (Semantic Transformer)")
        for movie in results:
            st.write(movie)

elif option == "Hybrid Movie Based":
    movie_name = st.selectbox("Select a movie", filtered_df['title'].sort_values())
    if st.button("Recommend"):
        results = hybrid_recommend(movie_name)
        st.subheader("Recommended Movies (Hybrid)")
        for movie in results:
            st.write(movie)

elif option == "Genre Based":
    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")
    if st.button("Recommend"):
        results = recommend_by_genre_from_tags(genre)
        st.subheader("Recommended Movies (Genre)")
        for movie in results:
            st.write(movie)

elif option == "Semantic Search":
    query = st.text_input("Type your search query (e.g., 'funny animated toy movie')")
    if st.button("Search"):
        query_emb = transformer_model.encode([query])
        sim_scores = cosine_similarity(query_emb, embeddings).flatten()
        top_idx = sim_scores.argsort()[::-1][:10]
        results = df['title'].iloc[top_idx].values
        st.subheader("Search Results (Semantic)")
        for movie in results:
            st.write(movie)
