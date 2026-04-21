"""
save_model.py  —  Trains the recommender and saves artefacts.
Run this ONCE before starting app.py.

Usage:
    python save_model.py
"""

import ast
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# ─── 1. LOAD DATA ───────────────────────────────────────────────────────────
# Put both CSV files in the same folder as this script.
movies  = pd.read_csv('tmdb_5000_movies.csv/tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv/tmdb_5000_credits.csv')

# ─── 2. MERGE ───────────────────────────────────────────────────────────────
movies = movies.merge(credits, on='title')
movies = movies[['id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]
movies.dropna(inplace=True)

# ─── 3. PARSE COLUMNS ───────────────────────────────────────────────────────
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)][:3]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['genres']   = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast']     = movies['cast'].apply(convert3)
movies['crew']     = movies['crew'].apply(fetch_director)
movies = movies.rename(columns={'crew': 'director'})

# ─── 4. BUILD TAGS ──────────────────────────────────────────────────────────
def clean(lst):
    return [i.replace(' ', '') for i in lst]

movies['overview']  = movies['overview'].apply(lambda x: x.split())
movies['genres']    = movies['genres'].apply(clean)
movies['keywords']  = movies['keywords'].apply(clean)
movies['cast']      = movies['cast'].apply(clean)
movies['director']  = movies['director'].apply(clean)

movies['tags'] = (
    movies['genres'] + movies['keywords'] +
    movies['overview'] + movies['cast'] + movies['director']
)

new_df = movies[['id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x).lower())

# ─── 5. STEMMING ────────────────────────────────────────────────────────────
ps = PorterStemmer()

def stem(text):
    return ' '.join(ps.stem(w) for w in text.split())

new_df['tags'] = new_df['tags'].apply(stem)

# ─── 6. VECTORISE & SIMILARITY ──────────────────────────────────────────────
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors    = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# ─── 7. SAVE ────────────────────────────────────────────────────────────────
pickle.dump(new_df,     open('movies.pkl',     'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print(f"✅  Saved movies.pkl ({len(new_df)} rows) and similarity.pkl.")
