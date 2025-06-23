import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the datasets
movies_df = pd.read_csv("C:/Users/shiva/OneDrive/Desktop/MinorProject/dataset/tmdb_5000_movies.csv")
credits_df = pd.read_csv("C:/Users/shiva/OneDrive/Desktop/MinorProject/dataset/tmdb_5000_credits.csv")


# Merge datasets on 'title'
movies_df = movies_df.merge(credits_df, on='title')

# Keep only necessary columns
movies_df = movies_df[['movie_id', 'title', 'overview']]

# Drop rows with missing overviews
movies_df.dropna(subset=['overview'], inplace=True)

# Vectorize the overview text
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['overview'])

# Compute cosine similarity
similarity = cosine_similarity(tfidf_matrix)

# Save DataFrame with title and id for lookup
movie_data = movies_df[['title', 'movie_id']].copy()
movie_data.rename(columns={'movie_id': 'id'}, inplace=True)



# Create the 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save to disk
joblib.dump(movie_data, "data/movies.pkl")
joblib.dump(similarity, "data/similarity.pkl")

print("✅ movies.pkl and similarity.pkl generated successfully.")