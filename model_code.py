# model_code.py
import pandas as pd
import numpy as np
import re
import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ------------------- NLTK setup ------------------- #
# Use a local folder for nltk data (put nltk_data folder in repo if possible)
nltk_data_path = "./nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Download only if missing
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', download_dir=nltk_data_path)
    stop_words = set(stopwords.words('english'))

try:
    _ = nltk.corpus.wordnet.ensure_loaded()
except:
    nltk.download('wordnet', download_dir=nltk_data_path)

lemmatizer = WordNetLemmatizer()

# ------------------- Text preprocessing ------------------- #
def process_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # remove stopwords & lemmatize
    return " ".join(words)

# ------------------- Genre parsing ------------------- #
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

# ------------------- Movie-based recommendation ------------------- #
def recommend(title, df, indices, tfidf_matrix, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

# ------------------- Semantic recommendation ------------------- #
def semantic_recommend(title, df, indices, embeddings, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    movie_emb = embeddings[idx]
    sim_scores = np.dot(embeddings, movie_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(movie_emb))
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx].values

# ------------------- Genre-based recommendation ------------------- #
def recommend_by_genre_from_tags(user_genre, df, n=10):
    user_genre = user_genre.lower().strip()
    genre_filtered_df = df[df['tags'].str.contains(user_genre, case=False, na=False)]
    if genre_filtered_df.empty:
        return ['Genre not found']
    return genre_filtered_df['title'].head(n).values
