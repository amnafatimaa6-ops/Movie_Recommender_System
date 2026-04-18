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

# ------------------ APP ------------------ #
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# ------------------ DATA ------------------ #
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

# ------------------ MODEL ------------------ #
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

# ------------------ TRAILER (REAL FIX) ------------------ #
def get_trailer_embed(title):
    try:
        query = title + " official trailer"
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"

        html = requests.get(url, timeout=5).text
        video_ids = re.findall(r"watch\?v=(.{11})", html)

        if video_ids:
            return f"https://www.youtube.com/embed/{video_ids[0]}"

        return None

    except:
        return None

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

# ------------------ MAIN ------------------ #
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

            st.markdown(f"## 🎬 {row['title']}")

            col1, col2 = st.columns([2,1])

            with col1:
                st.write(f"⭐ Rating: {row['vote_average']}")
                st.write(f"🔥 Popularity: {row['popularity']:.2f}")
                st.write(f"📊 Similarity: {row['similarity']:.2f}")
                st.write("📝 Overview:")
                st.write(row['overview'])

            with col2:
                st.markdown("### 🎥 Trailer")

                embed_url = get_trailer_embed(row['title'])

                if embed_url:
                    st.components.v1.html(
                        f"""
                        <iframe width="100%" height="420"
                        src="{embed_url}"
                        frameborder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowfullscreen>
                        </iframe>
                        """,
                        height=450
                    )
                else:
                    st.write("Trailer not found")

            st.divider()

# ------------------ GENRE ------------------ #
elif option == "Genre Based":

    genre = st.text_input("Enter genre")

    if st.button("Recommend"):

        res = df[df['tags'].str.contains(genre.lower(), na=False)].head(10)

        for _, row in res.iterrows():

            st.markdown(f"## 🎬 {row['title']}")
            st.write(f"⭐ {row['vote_average']}")
            st.write(row['overview'])

            embed_url = get_trailer_embed(row['title'])

            if embed_url:
                st.components.v1.html(
                    f"""
                    <iframe width="100%" height="420"
                    src="{embed_url}"
                    frameborder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowfullscreen>
                    </iframe>
                    """,
                    height=450
                )

            else:
                st.write("Trailer not found")

            st.divider()
