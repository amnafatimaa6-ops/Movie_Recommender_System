import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from model_code import recommend, recommend_by_genre_from_tags

# Load dataset
df = pd.read_csv("clean_movies.csv")

# Create title index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Build TF-IDF matrix from tags column
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

st.title("🎬 Movie Recommendation System")

option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("Movie Based", "Genre Based")
)

# Movie recommendation
if option == "Movie Based":
    movie_name = st.text_input("Enter movie name")

    if st.button("Recommend"):
        results = recommend(movie_name, df, indices, tfidf_matrix)

        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)

# Genre recommendation
elif option == "Genre Based":
    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")

    if st.button("Recommend"):
        results = recommend_by_genre_from_tags(genre, df)

        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)
