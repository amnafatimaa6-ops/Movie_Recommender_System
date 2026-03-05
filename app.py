# app.py
import streamlit as st
from movie_recommender import fetch_movie, fetch_movies_by_genre, recommend_similar_movies

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation App")

option = st.sidebar.selectbox("Choose Recommendation Type", ["Movie Based", "Genre Based"])

if option == "Movie Based":
    movie_name = st.text_input("Type a movie name")
    if st.button("Recommend Movies"):
        if not movie_name:
            st.warning("Type a movie first!")
        else:
            recs = recommend_similar_movies(movie_name)
            if not recs:
                st.info("No similar movies found.")
            else:
                cols = st.columns(3)
                for i, movie in enumerate(recs):
                    poster = movie.get('Poster')
                    with cols[i % 3]:
                        if poster != "N/A":
                            st.image(poster)
                        st.caption(f"{movie['Title']} ({movie.get('Year','')})")

elif option == "Genre Based":
    genre_name = st.text_input("Type a genre (e.g., Action, Comedy, Drama)")
    if st.button("Show Movies"):
        if not genre_name:
            st.warning("Type a genre first!")
        else:
            recs = fetch_movies_by_genre(genre_name)
            if not recs:
                st.info("No movies found in this genre.")
            else:
                cols = st.columns(3)
                for i, movie in enumerate(recs):
                    poster = movie.get('Poster')
                    with cols[i % 3]:
                        if poster != "N/A":
                            st.image(poster)
                        st.caption(f"{movie['Title']} ({movie.get('Year','')})")
