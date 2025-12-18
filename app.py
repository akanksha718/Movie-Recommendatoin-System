import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import ast

st.title('Movie Recommender System')
st.markdown("Find similar movies based on genres, cast, keywords, and plot overview.")

# Cache data loading
@st.cache_data
def load_data():
    try:
        # Load CSV files
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
        # Merge on title
        movies = movies.merge(credits, on='title')
        
        # Select relevant columns
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        
        # Drop nulls
        movies.dropna(inplace=True)
        
        return movies
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Parse JSON-like strings
def convert(obj):
    L = []
    for i in obj:
        L.append(i['name'])
    return L

def convert_cast(obj):
    L = []
    count = 0
    for i in obj:
        if count != 3:
            L.append(i['name'])
            count += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Cache preprocessing
@st.cache_data
def preprocess_data(movies):
    try:
        # Parse genres
        movies['genres'] = movies['genres'].apply(ast.literal_eval)
        movies['genres'] = movies['genres'].apply(convert)
        
        # Parse keywords
        movies['keywords'] = movies['keywords'].apply(ast.literal_eval)
        movies['keywords'] = movies['keywords'].apply(convert)
        
        # Parse cast
        movies['cast'] = movies['cast'].apply(ast.literal_eval)
        movies['cast'] = movies['cast'].apply(convert_cast)
        
        # Parse crew and extract director
        movies['crew'] = movies['crew'].apply(fetch_director)
        
        # Split overview and remove spaces from names
        movies['overview'] = movies['overview'].apply(lambda x: x.split())
        movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
        
        # Combine features into tags
        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        
        # Convert tags to string
        movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
        
        # Lowercase
        movies['tags'] = movies['tags'].apply(lambda x: x.lower())
        
        # Stemming
        ps = PorterStemmer()
        movies['tags'] = movies['tags'].apply(lambda text: " ".join([ps.stem(word) for word in text.split()]))
        
        return movies
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

# Cache vectorization
@st.cache_data
def vectorize_data(movies):
    try:
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies['tags']).toarray()
        similarity = cosine_similarity(vectors)
        return similarity
    except Exception as e:
        st.error(f"Error vectorizing data: {e}")
        return None

# Recommendation function
def recommend(movie_title, movies, similarity, top_k=5):
    try:
        # Find movie index (case-insensitive)
        movie_matches = movies[movies['title'].str.lower() == movie_title.lower()]
        
        if movie_matches.empty:
            return None
        
        movie_index = movie_matches.index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_k+1]
        
        recommendations = []
        for i, score in movies_list:
            recommendations.append({
                'title': movies.iloc[i]['title'],
                'similarity': f"{score:.2%}"
            })
        
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return None

# Main app logic
movies = load_data()

if movies is not None:
    # Preprocess data
    movies = preprocess_data(movies)
    
    if movies is not None:
        # Vectorize data
        similarity = vectorize_data(movies)
        
        if similarity is not None:
            # User input
            st.sidebar.header("Select a Movie")
            movie_list = movies['title'].values.tolist()
            selected_movie = st.sidebar.selectbox("Choose a movie:", movie_list)
            
            # Number of recommendations
            top_k = st.sidebar.slider("Number of recommendations:", 1, 10, 5)
            
            # Generate recommendations
            if st.sidebar.button("Get Recommendations"):
                recommendations = recommend(selected_movie, movies, similarity, top_k)
                
                if recommendations:
                    st.subheader(f"Movies similar to '{selected_movie}':")
                    for idx, rec in enumerate(recommendations, 1):
                        st.write(f"{idx}. **{rec['title']}** (Similarity: {rec['similarity']})")
                else:
                    st.error("No recommendations found.")
else:
    st.error("Could not load data. Make sure CSV files are in the correct directory.")