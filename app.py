import streamlit as st
from movie_recommender import recommend_movies, recommend_by_genre, get_movie_list, get_genre_list

# --- Page config ---
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# --- Sidebar: Select mode ---
st.sidebar.header("Choose Recommendation Mode")
mode = st.sidebar.radio("Recommend by:", ["Movie", "Genre"])

if mode == "Movie":
    # Recommend by movie
    favorite_movie = st.sidebar.selectbox("Pick a movie you like:", get_movie_list())
    num_recs = st.sidebar.slider("Number of recommendations:", 1, 10, 5)
    
    if st.sidebar.button("Recommend"):
        recommendations = recommend_movies(favorite_movie, num_recs)
        st.subheader("We recommend based on your movie:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

else:
    # Recommend by genre
    genre = st.sidebar.selectbox("Pick a genre:", get_genre_list())
    num_recs = st.sidebar.slider("Number of recommendations:", 1, 10, 5)
    
    if st.sidebar.button("Recommend by Genre"):
        recommendations = recommend_by_genre(genre, num_recs)
        st.subheader(f"We recommend from {genre} movies:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
