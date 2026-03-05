# movie_recommender.py
import requests

OMDB_API_KEY = "1d31e725"

def fetch_movie(title):
    """Fetch movie details by title"""
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    r = requests.get(url).json()
    if r.get('Response') == 'True':
        return r
    return None

def fetch_movies_by_genre(genre, n=10):
    """Fetch popular movies by genre (searching top titles)"""
    search_titles = ["Avengers", "Batman", "Spider-Man", "Frozen", "Inception",
                     "Titanic", "Joker", "Star Wars", "Lord of the Rings", "Toy Story"]
    results = []
    for t in search_titles:
        movie = fetch_movie(t)
        if movie and genre.lower() in movie.get('Genre', '').lower():
            results.append(movie)
        if len(results) >= n:
            break
    return results

def recommend_similar_movies(title):
    """Recommend movies similar by title keywords"""
    keywords = title.split()[:2]  # take first 1–2 words
    search_titles = ["Avengers", "Batman", "Spider-Man", "Frozen", "Inception",
                     "Titanic", "Joker", "Star Wars", "Lord of the Rings", "Toy Story"]
    results = []
    for t in search_titles:
        movie = fetch_movie(t)
        if movie:
            for kw in keywords:
                if kw.lower() in movie.get('Title','').lower() and movie['Title'].lower() != title.lower():
                    results.append(movie)
    return results
