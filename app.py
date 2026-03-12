import streamlit as st
import pandas as pd
import numpy as np
import re, ast
import nltk
import gdown

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

st.title("🎬 Movie Recommendation System")

# ---------------- DOWNLOAD DATASET ---------------- #
file_id = "1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"
url = f"https://drive.google.com/uc?id={file_id}"

csv_path = "movies_metadata.csv"
gdown.download(url, csv_path, quiet=True)

df = pd.read_csv(csv_path, low_memory=False)

# ---------------- CLEAN DATA ---------------- #
df = df[['title','overview','genres','tagline']]
df = df.dropna(subset=['title'])

df['overview'] = df['overview'].fillna("")
df['tagline'] = df['tagline'].fillna("")

df = df.reset_index(drop=True)

# ---------------- NLTK ---------------- #
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------- GENRE PARSER ---------------- #
def parse_genres(x):
    try:
        data = ast.literal_eval(x)
        return " ".join([i['name'] for i in data])
    except:
        return ""

# ---------------- TEXT CLEANING ---------------- #
def process_text(text):

    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words
    ]

    return " ".join(words)

df['genres'] = df['genres'].apply(parse_genres)

df['tags'] = (
    df['overview'] +
    " " +
    df['genres'] +
    " " +
    df['tagline']
).apply(process_text)

# ---------------- TF-IDF ---------------- #
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ---------------- TRANSFORMER MODEL ---------------- #
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data
def compute_embeddings(texts):

    emb = model.encode(texts.tolist())

    emb = np.asarray(emb).astype("float32")

    if emb.ndim == 1:
        emb = emb.reshape(-1,1)

    return emb

embeddings = compute_embeddings(df['tags'])

# ---------------- INDEX ---------------- #
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# ---------------- TFIDF RECOMMEND ---------------- #
def recommend(title, n=10):

    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]

    sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    )[0]

    sim[idx] = -1

    top_idx = np.argsort(sim)[::-1][:n]

    return df['title'].iloc[top_idx].values

# ---------------- SEMANTIC RECOMMEND ---------------- #
def semantic_recommend(title, n=10):

    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]

    movie_vec = embeddings[idx].reshape(1, -1)

    sim = cosine_similarity(movie_vec, embeddings)[0]

    sim[idx] = -1

    top_idx = np.argsort(sim)[::-1][:n]

    return df['title'].iloc[top_idx].values

# ---------------- HYBRID RECOMMEND ---------------- #
def hybrid_recommend(title, n=10):

    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]

    tfidf_sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    )[0]

    movie_vec = embeddings[idx].reshape(1, -1)

    semantic_sim = cosine_similarity(
        movie_vec,
        embeddings
    )[0]

    combined = 0.4 * tfidf_sim + 0.6 * semantic_sim

    combined[idx] = -1

    top_idx = np.argsort(combined)[::-1][:n]

    return df['title'].iloc[top_idx].values

# ---------------- GENRE RECOMMEND ---------------- #
def recommend_by_genre(genre):

    genre = genre.lower()

    results = df[df['tags'].str.contains(genre)]

    if results.empty:
        return ["Genre not found"]

    return results['title'].head(10).values

# ---------------- STREAMLIT UI ---------------- #

option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    [
        "TF-IDF Movie Based",
        "Semantic Movie Based",
        "Hybrid Movie Based",
        "Genre Based"
    ]
)

if option != "Genre Based":

    movie_name = st.selectbox(
        "Select a movie",
        df['title'].sort_values()
    )

    if st.button("Recommend"):

        if option == "TF-IDF Movie Based":

            results = recommend(movie_name)
            st.subheader("Recommended Movies (TF-IDF)")

        elif option == "Semantic Movie Based":

            results = semantic_recommend(movie_name)
            st.subheader("Recommended Movies (Semantic)")

        else:

            results = hybrid_recommend(movie_name)
            st.subheader("Recommended Movies (Hybrid AI)")

        for movie in results:
            st.write(movie)

else:

    genre = st.text_input("Enter genre")

    if st.button("Recommend"):

        results = recommend_by_genre(genre)

        st.subheader("Recommended Movies")

        for movie in results:
            st.write(movie)
