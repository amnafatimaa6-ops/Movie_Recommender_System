import streamlit as st
import pandas as pd
import numpy as np
import re, ast, os, pickle
import urllib.parse
import requests
import nltk
import gdown

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ------------------ APP SETUP ------------------ #
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# ------------------ DATA LOAD ------------------ #
file_id = "1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"
url = f"https://drive.google.com/uc?id={file_id}"
csv_path = "movies_metadata.csv"

if not os.path.exists(csv_path):
    gdown.download(url, csv_path, quiet=False)

df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8", on_bad_lines="skip")

df = df[['title','overview','genres','tagline','vote_average','popularity']]
df = df.drop_duplicates(subset='title').reset_index(drop=True)
df = df.dropna(subset=['title'])

df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0.0)

# ------------------ NLP ------------------ #
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

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# ------------------ TF-IDF ------------------ #
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ------------------ TRANSFORMER ------------------ #
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

embedding_file = "embeddings.pkl"

if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
else:
    embeddings = model.encode(df['tags'].tolist())
    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings, f)

# ------------------ 🌍 SAFE WIKIPEDIA ------------------ #
def get_movie_details(title):

    try:
        clean_title = urllib.parse.quote(title)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_title}"

        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            raise Exception("No page")

        res = response.json()

        poster = res.get("thumbnail", {}).get("source")
        summary = res.get("extract")
        wiki = res.get("content_urls", {}).get("desktop", {}).get("page")

        if not poster:
            poster = f"https://source.unsplash.com/300x450/?movie,{title}"

        return {
            "poster": poster,
            "summary": summary,
            "wiki": wiki
        }

    except:
        return {
            "poster": f"https://source.unsplash.com/300x450/?movie,{title}",
            "summary": None,
            "wiki": None
        }

# ------------------ 🎥 TRAILER (NO API) ------------------ #
def get_trailer_link(title):
    query = urllib.parse.quote_plus(title + " official trailer")
    return f"https://www.youtube.com/results?search_query={query}"

# ------------------ RECOMMENDERS ------------------ #
def recommend(title, n=10):
    idx = indices[title]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top = sim.argsort()[::-1][1:n+1]
    return df.iloc[top].assign(similarity=sim[top])

def semantic_recommend(title, n=10):
    idx = indices[title]
    sim = cosine_similarity(embeddings[idx].reshape(1,-1), embeddings).flatten()
    top = sim.argsort()[::-1][1:n+1]
    return df.iloc[top].assign(similarity=sim[top])

def hybrid_recommend(title, n=10):
    idx = indices[title]
    tf = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sem = cosine_similarity(embeddings[idx].reshape(1,-1), embeddings).flatten()
    combined = 0.5*tf + 0.5*sem
    top = combined.argsort()[::-1][1:n+1]
    return df.iloc[top].assign(similarity=combined[top])

# ------------------ UI ------------------ #
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("TF-IDF Movie Based", "Semantic Movie Based", "Hybrid Recommendation", "Genre Based")
)

# ------------------ MOVIE MODE ------------------ #
if option in ["TF-IDF Movie Based", "Semantic Movie Based", "Hybrid Recommendation"]:

    movie_name = st.selectbox("Select a movie", df['title'].sort_values())

    if st.button("Recommend"):

        if option == "TF-IDF Movie Based":
            results = recommend(movie_name)
            st.subheader("TF-IDF Recommendations")

        elif option == "Semantic Movie Based":
            results = semantic_recommend(movie_name)
            st.subheader("Semantic Recommendations")

        else:
            results = hybrid_recommend(movie_name)
            st.subheader("Hybrid Recommendations")

        for _, row in results.iterrows():

            data = get_movie_details(row['title'])

            col1, col2 = st.columns([1,2])

            with col1:
                st.image(data["poster"], width=160)

            with col2:
                st.markdown(f"### 🎬 {row['title']}")
                st.write(f"⭐ Rating: {row['vote_average']}")
                st.write(f"🔥 Popularity: {row['popularity']:.2f}")
                st.write(f"📊 Similarity: {row['similarity']:.2f}")

                if data["summary"]:
                    st.write(data["summary"])

                if data["wiki"]:
                    st.link_button("📖 Wikipedia", data["wiki"])

                st.link_button("▶ Watch Trailer", get_trailer_link(row['title']))

            st.divider()

# ------------------ GENRE MODE ------------------ #
elif option == "Genre Based":

    genre = st.text_input("Enter genre")

    if st.button("Recommend"):

        res = df[df['tags'].str.contains(genre.lower(), na=False)].head(10)

        for _, row in res.iterrows():

            data = get_movie_details(row['title'])

            st.markdown(f"### 🎬 {row['title']}")
            st.write(f"⭐ {row['vote_average']}")

            st.image(data["poster"], width=160)
            st.write(row['overview'])

            st.link_button("▶ Watch Trailer", get_trailer_link(row['title']))
            st.divider()
