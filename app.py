# ------------------- Semantic Transformer Recommendation ------------------- #
from sentence_transformers import SentenceTransformer
import numpy as np

# Load transformer model
@st.cache_resource(show_spinner=True)
def load_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_transformer_model()

# Compute embeddings for all movie tags
@st.cache_data(show_spinner=True)
def compute_embeddings(tags):
    return model.encode(tags.tolist(), show_progress_bar=True)

embeddings = compute_embeddings(df['tags'])

# Semantic recommendation function
def semantic_recommend(title, df, indices, embeddings, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    movie_emb = embeddings[idx]
    sim_scores = np.dot(embeddings, movie_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(movie_emb))
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

# ------------------- Update Streamlit UI ------------------- #
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("Movie Based", "Genre Based", "Semantic Movie Based")  # Added semantic option
)

if option == "Movie Based":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    if st.button("Recommend"):
        results = recommend(movie_name, df, indices, tfidf_matrix)
        st.subheader("Recommended Movies (TF-IDF)")
        for movie in results:
            st.write(movie)

elif option == "Semantic Movie Based":
    movie_name = st.selectbox("Select a movie", df['title'].sort_values())
    if st.button("Recommend"):
        results = semantic_recommend(movie_name, df, indices, embeddings)
        st.subheader("Recommended Movies (Semantic Transformer)")
        for movie in results:
            st.write(movie)

elif option == "Genre Based":
    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")
    if st.button("Recommend"):
        results = recommend_by_genre_from_tags(genre, df)
        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)
