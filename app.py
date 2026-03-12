import streamlit as st
import pandas as pd
import numpy as np
import re, ast, os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gdown

st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

# ------------------- Google Drive CSV ------------------- #
file_id = "1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"
url = f"https://drive.google.com/uc?id={file_id}"
csv_path = "movies_metadata.csv"

if not os.path.exists(csv_path):
    gdown.download(url, csv_path, quiet=False)

# ------------------- Load CSV ------------------- #
df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8", on_bad_lines="skip")

# ------------------- Clean & prepare ------------------- #
df = df.drop_duplicates().reset_index(drop=True)
df = df[['title','overview','genres','tagline','vote_average','popularity']]
df = df.dropna(subset=['title'])
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')

# NLTK setup
import nltk
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

# ------------------- Build Index & TF-IDF ------------------- #
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ------------------- Transformer Model & Embeddings ------------------- #
model_path = "sentence_model"
embeddings_path = "movie_embeddings.npy"

@st.cache_resource(show_spinner=True)
def load_transformer():
    if os.path.exists(model_path):
        return SentenceTransformer(model_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save(model_path)
    return model

transformer_model = load_transformer()

@st.cache_data(show_spinner=True)
def get_embeddings(tags_list):
    if os.path.exists(embeddings_path):
        return np.load(embeddings_path)
    embs = transformer_model.encode(tags_list, show_progress_bar=True)
    np.save(embeddings_path, embs)
    return embs

embeddings = get_embeddings(df['tags'].tolist())

# ------------------- Recommendation Functions ------------------- #
def recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

def semantic_recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    movie_emb = embeddings[idx].reshape(1, -1)
    sim_scores = cosine_similarity(movie_emb, embeddings).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

def hybrid_recommend(title, n=10, alpha=0.5):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]

    # TF-IDF
    tfidf_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    # Transformer
    movie_emb = embeddings[idx].reshape(1, -1)
    semantic_sim = cosine_similarity(movie_emb, embeddings).flatten()
    # Weighted hybrid
    hybrid_score = alpha*tfidf_sim + (1-alpha)*semantic_sim
    similar_idx = hybrid_score.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

def recommend_by_genre_from_tags(user_genre, n=10):
    user_genre = user_genre.lower().strip()
    genre_filtered_df = df[df['tags'].str.contains(user_genre, case=False, na=False)]
    if genre_filtered_df.empty:
        return ['Genre not found']
    return genre_filtered_df['title'].head(n).values

# ------------------- Streamlit UI ------------------- #
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("TF-IDF Movie Based", "Semantic Movie Based", "Hybrid Movie Based", "Genre Based")
)

if option == "TF-IDF Movie Based":
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

elif option == "Hybrid Movie Based":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    if st.button("Recommend"):
        results = hybrid_recommend(movie_name)
        st.subheader("Recommended Movies (Hybrid TF-IDF + Semantic)")
        for movie in results:
            st.write(movie)

elif option == "Genre Based":
    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")
    if st.button("Recommend"):
        results = recommend_by_genre_from_tags(genre)
        st.subheader("Recommended Movies by Genre")
        for movie in results:
            st.write(movie)
