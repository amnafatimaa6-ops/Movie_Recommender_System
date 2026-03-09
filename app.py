import streamlit as st
import pandas as pd
import numpy as np

from model_code import recommend, semantic_recommend, recommend_by_genre_from_tags

# Load cleaned dataset
df = pd.read_csv("clean_movies.csv")

# Title index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Load saved matrices
tfidf_matrix = np.load("tfidf_matrix.npy")
embeddings = np.load("embeddings.npy")

st.title("🎬 Movie Recommendation System")

option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("Movie Based", "Semantic", "Genre Based")
)

# ---------------- Movie Based ----------------
if option == "Movie Based":
    movie_name = st.text_input("Enter movie name")

    if st.button("Recommend"):
        results = recommend(movie_name, df, indices, tfidf_matrix)

        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)

# ---------------- Semantic ----------------
elif option == "Semantic":
    movie_name = st.text_input("Enter movie name")

    if st.button("Recommend"):
        results = semantic_recommend(movie_name, df, indices, embeddings)

        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)

# ---------------- Genre Based ----------------
elif option == "Genre Based":
    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")

    if st.button("Recommend"):
        results = recommend_by_genre_from_tags(genre, df)

        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)
