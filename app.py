import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gdown  # to download from Google Drive

from model_code import recommend, recommend_by_genre_from_tags

st.title("🎬 Movie Recommendation System")

# ------------------- Google Drive CSV ------------------- #
# Google Drive file ID
file_id = "1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm"  # your link: https://drive.google.com/file/d/1KdZYGA_gR3Cip09HvwYZf7gGi6aQY6rm/view?usp=sharing
url = f"https://drive.google.com/uc?id={file_id}"

# Download CSV to local cache
csv_path = "movies_metadata.csv"
gdown.download(url, csv_path, quiet=False)

# ------------------- Load CSV ------------------- #
df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8", on_bad_lines="skip")

# ------------------- Clean & prepare for recommendation ------------------- #
# Drop duplicates and select relevant columns
df = df.drop_duplicates().reset_index(drop=True)
df = df[['title','overview','genres','tagline','vote_average','popularity']]
df = df.dropna(subset=['title'])
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')

# Preprocess genres and create tags
import ast, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
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
df['tags'] = (df['overview'] + ' ' + df['genres'] + ' ' + df['tagline']).apply(process_text)

# Build title index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# TF-IDF matrix
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ------------------- Streamlit UI ------------------- #
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ("Movie Based", "Genre Based")
)

if option == "Movie Based":
    movie_name = st.text_input("Enter movie name")
    if st.button("Recommend"):
        results = recommend(movie_name, df, indices, tfidf_matrix)
        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)

elif option == "Genre Based":
    genre = st.text_input("Enter genre (Action, Comedy, Horror etc)")
    if st.button("Recommend"):
        results = recommend_by_genre_from_tags(genre, df)
        st.subheader("Recommended Movies")
        for movie in results:
            st.write(movie)
