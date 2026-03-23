from datetime import datetime

from flask import Flask, render_template, request
import pandas as pd
import ast
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import requests
from dotenv import load_dotenv
import os

# --- Load API key ---
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not set in .env file")

# --- Load datasets ---
movies = pd.read_csv('datasets/tmdb_5000_movies.csv')
credits = pd.read_csv('datasets/tmdb_5000_credits.csv')

credits = credits.rename(columns={'movie_id': 'id'})
movies = movies.merge(credits, on='id')

if 'title_x' in movies.columns:
    movies = movies.rename(columns={'title_x': 'title'})
if 'title_y' in movies.columns:
    movies = movies.drop(columns=['title_y'])

movies['genres'] = movies['genres'].fillna('')
movies['cast'] = movies['cast'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')
movies['overview'] = movies['overview'].fillna('')

print(movies.columns)

# --- Helper functions ---
def extract_names(data):
    try:
        if isinstance(data, str):
            data = ast.literal_eval(data)
        if isinstance(data, list):
            return " ".join([item['name'] for item in data if isinstance(item, dict)])
    except:
        return ""
    return ""

def combine_features(row):
    return (
        extract_names(row['genres']) + " " +
        extract_names(row['keywords']) + " " +
        extract_names(row['cast']) + " " +
        str(row['overview'])
    )

movies['combined_features'] = movies.apply(combine_features, axis=1)

# --- TF-IDF ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# --- Fetch poster ---
def get_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except:
        pass
    return "https://via.placeholder.com/300x450?text=No+Image"

# --- Fetch ALL matches from TMDB ---
def fetch_all_movies(movie_name, release_year=None, language=None):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}"
    data = requests.get(url).json()

    language_map = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese"
    }
    if not data.get('results'):
        return []

    results = data['results']

    # Filter only released movies
    today = datetime.today().strftime('%Y-%m-%d')

    results = [
        m for m in results
        if m.get('release_date') and m.get('release_date') <= today
    ]

    # Filter year
    if release_year:
        results = [m for m in results if m.get('release_date', '').startswith(str(release_year))]

    # Filter language
    if language:
        results = [m for m in results if m.get('original_language') == language]

    # 🔥 STRICT TITLE MATCH FIRST
    exact = [m for m in results if movie_name.lower() == m['title'].lower()]
    if exact:
        results = exact

    # Sort by popularity
    results.sort(key=lambda x: x.get('popularity', 0), reverse=True)

    return [
        {
            "title": m['title'],
            "id": m['id'],
            "poster": get_poster(m.get('id')),
            "release_date": m.get('release_date'),
            "language": language_map.get(m.get('original_language'), m.get('original_language'))
        }
        for m in results[:10]
    ]

# --- Fetch recommendations ---
def fetch_similar_movies(movie_id, top_n=5, release_year=None, language=None):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={api_key}"
    data = requests.get(url).json()

    language_map = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese"
    }

    results = data.get('results', [])

    # Filter year
    if release_year:
        results = [m for m in results if m.get('release_date', '').startswith(str(release_year))]

    # Filter language
    if language:
        results = [m for m in results if m.get('original_language') == language]

    return [
        {
            "title": m['title'],
            "poster": get_poster(m.get('id')),
            "release_date": m.get('release_date'),
            "language": language_map.get(m.get('original_language'), m.get('original_language'))
        }
        for m in results[:top_n]
    ]

# --- Fuzzy match local dataset ---
def get_movie_index(movie_title):
    best_match, score = process.extractOne(movie_title, movies['title'].values)

    print("User Input:", movie_title)
    print("Matched With:", best_match)
    print("Score:", score)

    if score < 85:
        return None

    if movie_title.lower() not in best_match.lower():
        return None

    return movies[movies['title'] == best_match].index[0]

# --- Recommendation logic ---
def recommend_movies(movie_title, top_n, release_year=None, language=None):
    index = get_movie_index(movie_title)

    language_map = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese"
    }
    
    if index is None:
        print("➡ Using TMDB")

        matches = fetch_all_movies(movie_title)

        if not matches:
            return []

        return {"matches": matches}

    print("➡ Using TF-IDF")

    distances, indices = NearestNeighbors(n_neighbors=top_n+1, metric='cosine') \
        .fit(tfidf_matrix).kneighbors(tfidf_matrix[index])

    recommendations = []
    for i in indices[0][1:]:
        movie = movies.iloc[i]
        movie_id = movie['id']  # get the TMDB movie ID
        recommendations.append({
            "title": movie['title'],
            "poster": get_poster(movie_id),
            "release_date": movie.get('release_date'),
            "language": language_map.get(movie.get('original_language'), movie.get('original_language'))
        })

    return {"recommendations": recommendations}


# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    num_recommendations = int(request.form.get('num_recommendations', 4))  # Default to 4 if not provided
    result = recommend_movies(movie_title, top_n=num_recommendations)

    if not result:
        return render_template('index.html', error="Movie not found.")

    if "matches" in result:
        return render_template('select_movie.html', matches=result["matches"])

    return render_template('recommendations.html', recommendations=result["recommendations"], movie_title=movie_title)

@app.route('/recommend_by_id', methods=['POST'])
def recommend_by_id():
    movie_id = int(request.form['movie_id'])
    num = int(request.form.get('num_recommendations', 5))
    recs = fetch_similar_movies(movie_id, top_n=num)
    movie_title = request.form.get('movie_title')
    return render_template('recommendations.html', recommendations=recs, movie_title=movie_title)

if __name__ == '__main__':
    app.run(debug=True)