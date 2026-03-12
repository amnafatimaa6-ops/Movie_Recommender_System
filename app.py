import streamlit as st
import pandas as pd
import numpy as np
import ast, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sentence_transformers import SentenceTransformer
import gdown
import os

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

# ------------------- Clean & Prepare ------------------- #
df = df.drop_duplicates().reset_index(drop=True)
df = df[['title','overview','genres','tagline','vote_average','popularity']].dropna(subset=['title'])
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Genre parsing
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

# Text preprocessing
def process_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

df['genres'] = df['genres'].apply(parse_genres)
df['tags'] = (df['overview'] + ' ' + df['genres'] + ' ' + df['tagline']).apply(process_text)

# Build title index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# ------------------- TF-IDF ------------------- #
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ------------------- Semantic Transformer ------------------- #
@st.cache_resource(show_spinner=True)
def load_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

transformer_model = load_transformer_model()

@st.cache_data(show_spinner=True)
def get_embeddings(model):
    return model.encode(df['tags'].tolist(), show_progress_bar=True)

embeddings = get_embeddings(transformer_model)

# ------------------- Recommendation Functions ------------------- #
def recommend(title, n=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_idx = sim_scores.argsort()[::-1][1:n+1]
    return sim_idx

def semantic_recommend(title, n=10):
    if title not in indices:
        return []
    idx = indices[title]
    movie_emb = embeddings[idx].reshape(1, -1)
    sim_scores = cosine_similarity(movie_emb, embeddings)[0]
    sim_idx = sim_scores.argsort()[::-1][1:n+1]
    return sim_idx

def hybrid_recommend(title, n=10, alpha=0.5):
    if title not in indices:
        return []
    idx = indices[title]
    # TF-IDF
    tfidf_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    # Semantic
    movie_emb = embeddings[idx].reshape(1, -1)
    semantic_sim = cosine_similarity(movie_emb, embeddings)[0]
    # Weighted sum
    hybrid_sim = alpha*tfidf_sim + (1-alpha)*semantic_sim
    hybrid_idx = hybrid_sim.argsort()[::-1][1:n+1]
    return hybrid_idx

def recommend_by_genre_from_tags(user_genre, n=10):
    user_genre = user_genre.lower().strip()
    genre_filtered = df[df['tags'].str.contains(user_genre, case=False, na=False)]
    if genre_filtered.empty:
        return []
    return genre_filtered.head(n).index.tolist()

# ------------------- Pretty output ------------------- #
def get_movie_info(indices_list, method="TF-IDF", base_idx=None):
    rows = []
    for idx in indices_list:
        title = df['title'].iloc[idx]
        rating = df['vote_average'].iloc[idx]
        overview = df['overview'].iloc[idx]
        overview_short = (overview[:200] + '...') if len(overview) > 200 else overview

        # Similarity / explanation
        if base_idx is not None:
            if method in ["TF-IDF","Hybrid"]:
                sim_score = cosine_similarity(tfidf_matrix[base_idx], tfidf_matrix[idx])[0][0]
            elif method=="Semantic":
                sim_score = cosine_similarity(embeddings[base_idx].reshape(1,-1), embeddings[idx].reshape(1,-1))[0][0]
            else:
                sim_score = 0
            explanation = f"{method} similarity: {sim_score:.2f}"
        else:
            explanation = "N/A"

        rows.append({
            "Title": title,
            "Rating": rating,
            "Summary": overview_short,
            "Why": explanation
        })
    return pd.DataFrame(rows)

# ------------------- Streamlit UI ------------------- #
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("TF-IDF Movie Based", "Semantic Movie Based", "Hybrid Movie Based", "Genre Based")
)

if option == "TF-IDF Movie Based":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    if st.button("Recommend"):
        recommended_idx = recommend(movie_name)
        base_idx = indices[movie_name]
        pretty_df = get_movie_info(recommended_idx, method="TF-IDF", base_idx=base_idx)
        st.table(pretty_df)

elif option == "Semantic Movie Based":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    if st.button("Recommend"):
        recommended_idx = semantic_recommend(movie_name)
        base_idx = indices[movie_name]
        pretty_df = get_movie_info(recommended_idx, method="Semantic", base_idx=base_idx)
        st.table(pretty_df)

elif option == "Hybrid Movie Based":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    if st.button("Recommend"):
        recommended_idx = hybrid_recommend(movie_name)
        base_idx = indices[movie_name]
        pretty_df = get_movie_info(recommended_idx, method="Hybrid", base_idx=base_idx)
        st.table(pretty_df)

elif option == "Genre Based":
    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")
    if st.button("Recommend"):
        recommended_idx = recommend_by_genre_from_tags(genre)
        pretty_df = get_movie_info(recommended_idx, method="Genre")
        st.table(pretty_df)

# ------------------- Save TF-IDF and Transformer Model ------------------- #
# You can save TF-IDF and embeddings to avoid recomputation
import pickle
with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
np.save("transformer_embeddings.npy", embeddings)
