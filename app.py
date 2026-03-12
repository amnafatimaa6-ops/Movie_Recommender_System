import streamlit as st
import pandas as pd
import numpy as np
import re, ast, os, pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
import gdown

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# ------------------ Download Dataset ------------------ #
file_id = "1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"
url = f"https://drive.google.com/uc?id={file_id}"
csv_path = "movies_metadata.csv"

if not os.path.exists(csv_path):
    gdown.download(url, csv_path, quiet=False)

# ------------------ Load Dataset ------------------ #
df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8", on_bad_lines="skip")

# Remove duplicates
df = df.drop_duplicates(subset=['title']).reset_index(drop=True)

df = df[['title','overview','genres','tagline','vote_average','popularity']]

df = df.dropna(subset=['title'])

df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')
df['popularity'] = df['popularity'].fillna(0)
df['vote_average'] = df['vote_average'].fillna(0)

# ------------------ NLTK Setup ------------------ #
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------ Text Processing ------------------ #
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

df['tags'] = (
    df['overview'] + " " +
    df['genres'] + " " +
    df['tagline']
).apply(process_text)

# ------------------ TF-IDF ------------------ #
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ------------------ Transformer Model ------------------ #
@st.cache_resource(show_spinner=True)
def load_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

transformer_model = load_transformer_model()

# ------------------ Embedding Cache ------------------ #
embedding_file = "embeddings.pkl"

@st.cache_data(show_spinner=True)
def get_embeddings(tags_list):
    return transformer_model.encode(tags_list, show_progress_bar=True)

if os.path.exists(embedding_file):

    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)

else:

    embeddings = get_embeddings(df['tags'].tolist())

    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings, f)

# ------------------ Recommendation Functions ------------------ #
def recommend(title, n=10):

    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similar_idx = sim_scores.argsort()[::-1][1:n+1]

    return df.iloc[similar_idx][
        ['title','vote_average','overview','popularity']
    ].copy().assign(
        similarity=sim_scores[similar_idx]
    )

def semantic_recommend(title, n=10):

    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    movie_emb = embeddings[idx].reshape(1,-1)

    sim_scores = cosine_similarity(movie_emb, embeddings).flatten()

    similar_idx = sim_scores.argsort()[::-1][1:n+1]

    return df.iloc[similar_idx][
        ['title','vote_average','overview','popularity']
    ].copy().assign(
        similarity=sim_scores[similar_idx]
    )
def hybrid_recommend(title, n=10, tfidf_weight=0.5, semantic_weight=0.5):

    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    tfidf_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    movie_emb = embeddings[idx].reshape(1, -1)
    semantic_sim = cosine_similarity(movie_emb, embeddings).flatten()

    # Fix size mismatch
    min_len = min(len(tfidf_sim), len(semantic_sim))

    tfidf_sim = tfidf_sim[:min_len]
    semantic_sim = semantic_sim[:min_len]

    combined_sim = tfidf_weight * tfidf_sim + semantic_weight * semantic_sim

    similar_idx = combined_sim.argsort()[::-1][1:n+1]

    return df.iloc[similar_idx][
        ['title','vote_average','overview','popularity']
    ].copy().assign(
        similarity=combined_sim[similar_idx]
    )

def recommend_by_genre_from_tags(user_genre, n=10):

    user_genre = user_genre.lower().strip()

    genre_filtered_df = df[df['tags'].str.contains(user_genre, case=False, na=False)]

    if genre_filtered_df.empty:
        return pd.DataFrame()

    return genre_filtered_df.head(n)[
        ['title','vote_average','overview','popularity']
    ].copy().assign(similarity=1.0)

# ------------------ Streamlit UI ------------------ #
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    (
        "TF-IDF Movie Based",
        "Semantic Movie Based",
        "Hybrid Recommendation",
        "Genre Based"
    )
)

if option in [
    "TF-IDF Movie Based",
    "Semantic Movie Based",
    "Hybrid Recommendation"
]:

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
            st.subheader("Recommended Movies (Semantic Transformer)")

        else:

            results = hybrid_recommend(movie_name)
            st.subheader("Recommended Movies (Hybrid)")

        if results.empty:
            st.warning("No recommendations found.")
        else:

            avg_similarity = results['similarity'].mean()

            st.markdown(
                f"**Evaluation Metric:** Average similarity = {avg_similarity:.2f}"
            )

            for idx, row in results.iterrows():

                st.markdown(f"### 🎬 {row['title']}")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"⭐ **Rating:** {row['vote_average']}")
                    st.write(f"📊 **Similarity Score:** {row['similarity']:.2f}")

                with col2:
                    st.write(f"🔥 **Popularity:** {row['popularity']:.2f}")

                st.write("📝 **Overview:**")
                st.write(row['overview'])

                st.divider()

elif option == "Genre Based":

    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")

    if st.button("Recommend"):

        results = recommend_by_genre_from_tags(genre)

        st.subheader(f"Recommended Movies for Genre: {genre}")

        if results.empty:
            st.warning("No movies found for this genre.")

        else:
            for idx, row in results.iterrows():

                st.markdown(f"### 🎬 {row['title']}")
                st.write(f"⭐ **Rating:** {row['vote_average']}")
                st.write(f"🔥 **Popularity:** {row['popularity']}")
                st.write(row['overview'])

                st.divider()
