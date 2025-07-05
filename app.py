
import requests
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from io import BytesIO 
import time
import difflib
import speech_recognition as sr
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import os
import streamlit as st

# Force Streamlit to listen on all network interfaces
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_PORT"] = "8080"




# ------------------ Safe Request ------------------
def safe_get(url, max_retries=5, backoff_factor=1):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"[Retry {attempt + 1}] Connection failed: {e}. Retrying in {backoff_factor} seconds...")
            time.sleep(backoff_factor)
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            break
    return None

# ------------------ Load Models ------------------
preprocessor = joblib.load("models/preprocessor.pkl")
rating_model = joblib.load("models/rating_model.pkl")
revenue_model = joblib.load("models/revenue_model.pkl")
tokenizer = joblib.load("models/tokenizer.pkl")
dl_model = tf.keras.models.load_model(
    "models/overview_rating_model.h5",
    custom_objects={
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError(),
    }
)

# ------------------ TMDB API ------------------
API_KEY = "767e59e73de41cf06bc1133584eab132"


# ------------------ Get Top Movies by Year or Decade ------------------
def fetch_top_movies_by_year(year, language='en', count=10):
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&sort_by=vote_average.desc&primary_release_year={year}&vote_count.gte=100&with_original_language={language}"
    response = safe_get(url)
    if not response:
        return []
    return response.json().get("results", [])[:count]

def fetch_top_movies_by_decade(decade_start, language='en', count=10):
    decade_end = decade_start + 9
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&sort_by=vote_average.desc&primary_release_date.gte={decade_start}-01-01&primary_release_date.lte={decade_end}-12-31&vote_count.gte=100&with_original_language={language}"
    response = safe_get(url)
    if not response:
        return []
    return response.json().get("results", [])[:count]



# ------------------ Fetch Movie ------------------
def fetch_movie_details(movie_name):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    response = safe_get(search_url)
    if response is None:
        return None
    data = response.json()
    if data['results']:
        movie_id = data['results'][0]['id']
        detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits,videos,watch/providers,reviews"
        detail_response = safe_get(detail_url)
        return detail_response.json() if detail_response else None
    else:
        return None

# ------------------ Preprocessing ------------------
def preprocess_api_data(movie):
    try:
        budget = movie.get("budget", 0)
        runtime = movie.get("runtime", 0)
        popularity = movie.get("popularity", 0)
        overview = movie.get("overview", "")
        crew_list = movie.get("credits", {}).get("crew", [])
        director = next((i['name'] for i in crew_list if i['job'] == 'Director'), 'Unknown')

        X_ml = pd.DataFrame([{
            "budget": budget,
            "runtime": runtime,
            "popularity": popularity,
            "crew": director
        }])
        return X_ml, overview
    except:
        return None, None

# ------------------ Recommendation ------------------
movies_df = pd.read_csv("dataset/tmdb_5000_movies.csv")
similarity = joblib.load("data/similarity.pkl")

def fetch_movie_details_by_id(movie_id):
    detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits,videos,watch/providers,reviews"
    response = safe_get(detail_url)
    return response.json() if response else None

def recommend_movies(movie_name):
    movie_name = movie_name.lower()
    all_titles = movies_df['title'].str.lower().tolist()
    close_matches = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.6)
    
    if not close_matches:
        return []
    
    closest_title = close_matches[0]
    index = movies_df[movies_df['title'].str.lower() == closest_title].index[0]
    distances = similarity[index]
    top_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended = []
    for i in top_indices:
        movie_id = movies_df.iloc[i[0]]['id']
        recommended.append(fetch_movie_details_by_id(movie_id))
    return recommended

def recommend_from_tmdb_cast_crew(movie_input):
    movie = fetch_movie_details(movie_input)
    if not movie or 'credits' not in movie:
        return []

    cast_list = movie['credits']['cast'][:3]  # Top 3 actors
    crew_list = movie['credits']['crew']
    director = next((i for i in crew_list if i['job'] == 'Director'), None)

    # Search movies using actor or director
    recommendations = []
    used_ids = set()
    search_names = []

    if director:
        search_names.append(director['name'])
    search_names.extend([actor['name'] for actor in cast_list])

    for name in search_names:
        search_url = f"https://api.themoviedb.org/3/search/person?api_key={API_KEY}&query={name}"
        person_response = safe_get(search_url)
        if not person_response:
            continue

        person_data = person_response.json()
        if not person_data['results']:
            continue

        person_id = person_data['results'][0]['id']
        credits_url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?api_key={API_KEY}"
        credits_response = safe_get(credits_url)
        if not credits_response:
            continue

        credits = credits_response.json()
        for m in credits.get("cast", []):
            if m['id'] not in used_ids and m.get("poster_path"):
                used_ids.add(m['id'])
                recommendations.append(m)
                if len(recommendations) >= 5:
                    break
        if len(recommendations) >= 5:
            break

    return recommendations



    # Get cast/crew names
    cast = movie['credits'].get('cast', [])[:5]  # top 5 actors
    crew = movie['credits'].get('crew', [])
    director = next((member for member in crew if member.get('job') == 'Director'), None)

    similar_movies = set()

    # Recommend movies using top 5 cast members
    for person in cast:
        person_id = person.get("id")
        if person_id:
            url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?api_key={API_KEY}"
            response = safe_get(url)
            if response:
                data = response.json()
                for credit in data.get("cast", []):
                    if credit.get("id") != movie.get("id"):
                        similar_movies.add(credit["id"])

    # Add director's filmography
    if director:
        person_id = director.get("id")
        url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?api_key={API_KEY}"
        response = safe_get(url)
        if response:
            data = response.json()
            for credit in data.get("crew", []):
                if credit.get("job") == "Director" and credit.get("id") != movie.get("id"):
                    similar_movies.add(credit["id"])

    # Fetch top 5 movies with highest popularity
    movie_details_list = []
    for mid in list(similar_movies)[:20]:  # limit to avoid hitting API limits
        detail = fetch_movie_details_by_id(mid)
        if detail:
            movie_details_list.append(detail)

    sorted_by_popularity = sorted(movie_details_list, key=lambda x: x.get("popularity", 0), reverse=True)
    return sorted_by_popularity[:5]


# ------------------ Streamlit App ------------------




# Page Configuration
st.set_page_config(page_title="🎬 AI Movie Analyzer", layout="wide")
st.title("🎥 AI-Powered Movie Analyzer")
st.caption("🚀 Built with TMDB + ML + DL + NLP")
st.markdown("---")

# --- Sidebar Options ---
st.sidebar.header("🔧 Options Panel")
movie_input = st.sidebar.text_input("🔍 Enter Movie Name")

# Sidebar Options
option = st.sidebar.selectbox("Top Movies By Year & Decades ", ["Select", "Top Movies by Year", "Top Movies by Decade"])

# --- Top Movies by Year ---
if option == "Top Movies by Year":
    selected_year = st.sidebar.slider("Choose a Year", min_value=1950, max_value=2025, value=2010)
    industry = st.sidebar.radio("Choose Industry", ["Hollywood", "Bollywood"])
    language = 'en' if industry == "Hollywood" else 'hi'
    if st.sidebar.button("🎬 Show Top Movies for Selected Year"):
        with st.spinner("Fetching top movies..."):
            movies = fetch_top_movies_by_year(selected_year, language=language)
            if movies:
                st.subheader(f"🎯 Top {industry} Movies of {selected_year}")
                for m in movies:
                    st.image(f"https://image.tmdb.org/t/p/w200{m.get('poster_path')}", width=150)
                    st.markdown(f"**🎬 {m.get('title')}**")
                    st.markdown(f"📝 {m.get('overview', 'No overview available.')}") 
                    st.markdown("---")
            else:
                st.error("No data found.")

# --- Top Movies by Decade ---
elif option == "Top Movies by Decade":
    decade = st.sidebar.selectbox("Select Decade", [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
    industry = st.sidebar.radio("Choose Industry", ["Hollywood", "Bollywood"])
    language = 'en' if industry == "Hollywood" else 'hi'
    if st.sidebar.button("🎥 Show Top Movies for Selected Decade"):
        with st.spinner("Fetching top movies..."):
            movies = fetch_top_movies_by_decade(decade, language=language)
            if movies:
                st.subheader(f"📽 Top {industry} Movies from {decade}s")
                for m in movies:
                    st.image(f"https://image.tmdb.org/t/p/w200{m.get('poster_path')}", width=150)
                    st.markdown(f"**🎬 {m.get('title')}**")
                    st.markdown(f"📝 {m.get('overview', 'No overview available.')}") 
                    st.markdown("---")
            else:
                st.error("No data found.")

# --- Button Functionalities on Main Page ---
if st.button("🎯 Predict Rating"):
    if movie_input:
        with st.spinner("Fetching and predicting..."):
            movie = fetch_movie_details(movie_input)

            if not movie:
                st.error("❌ Movie not found.")
                st.stop()

            # ─── basic info ───
            st.subheader(f"🎞️ {movie['title']}")
            if movie.get("poster_path"):
                st.image(f"https://image.tmdb.org/t/p/w500{movie['poster_path']}")
            st.write(f"📅 **Release Date:** {movie.get('release_date','N/A')}")
            st.write(f"🎤 **Language:** {movie.get('original_language','N/A').upper()}")
            st.write(f"📝 **Overview:** {movie.get('overview','N/A')}")

            # ground‑truth values from TMDB
            actual_budget   = movie.get("budget",   0)
            actual_revenue  = movie.get("revenue",  0)

            # ─── prepare features for the ML model ───
            X_ml, overview  = preprocess_api_data(movie)     # your helper
            if X_ml is None:
                st.error("❌ Could not process data.")
                st.stop()

            pred_rating_ml  = rating_model.predict(X_ml)[0]
            pred_revenue_ml = revenue_model.predict(X_ml)[0]

            # ─── DL prediction from overview text ───
            seq   = tokenizer.texts_to_sequences([overview])
            pad   = pad_sequences(seq, maxlen=120, padding='post', truncating='post')
            pred_rating_dl = dl_model.predict(pad, verbose=0)[0][0]

            # ─── display results ───
            st.markdown("### 📊 Predicted Results")
            st.write(f"⭐ **Rating (ML):** {pred_rating_ml:.2f} / 10")
            st.write(f"🤖 **Rating (DL):** {pred_rating_dl:.2f} / 10")
            #st.write(f"💰 **Estimated Revenue:** ${int(pred_revenue_ml):,}")
            #st.write(f"🎯 **Actual Revenue (TMDB):** ${int(actual_revenue):,}")

            # budget vs revenue chart
            box_df = pd.DataFrame(
                {"Category": ["Budget", "Revenue"],
                 "Amount":   [actual_budget, actual_revenue]})
            st.subheader("💸 Box‑Office Comparison")
            st.bar_chart(box_df.set_index("Category"))
    else:
        st.warning("Enter a movie name.")

if st.button("🎬 Get Dataset-Based Recommendations"):
    if movie_input:
        with st.spinner("Generating dataset-based recommendations..."):
            recommended_movies = recommend_movies(movie_input)
            if recommended_movies:
                st.subheader("📽 Dataset-Based Recommendations:")
                for i in range(0, len(recommended_movies), 3):
                    row = st.columns(3)
                    for j in range(3):
                        if i + j < len(recommended_movies):
                            movie = recommended_movies[i + j]
                            with row[j]:
                                if movie:
                                    st.image(f"https://image.tmdb.org/t/p/w200{movie.get('poster_path')}", width=150)
                                    st.markdown(f"**🎬 {movie.get('title')}**")
                                    st.markdown(f"📝 {movie.get('overview', 'No overview available.')}")    
            else:
                st.warning("No recommendations found.")
    else:
        st.warning("Enter a movie name.")

if st.button("🌐 Get TMDB-Based Recommendations (Cast/Crew)"):
    if movie_input:
        with st.spinner("Fetching TMDB recommendations..."):
            tmdb_recs = recommend_from_tmdb_cast_crew(movie_input)
            if tmdb_recs:
                st.subheader("🌟 TMDB-Based Cast/Crew Recommendations:")
                for i in range(0, len(tmdb_recs), 3):
                    row = st.columns(3)
                    for j in range(3):
                        if i + j < len(tmdb_recs):
                            m = tmdb_recs[i + j]
                            with row[j]:
                                st.image(f"https://image.tmdb.org/t/p/w200{m.get('poster_path')}", width=150)
                                st.markdown(f"**🎬 {m.get('title')}**")
                                st.markdown(f"📝 {m.get('overview', 'No overview available.')}")    
                                st.caption(f"📅 {m.get('release_date', 'N/A')}")
            else:
                st.warning("No cast/crew-based recommendations found.")
    else:
        st.warning("Enter a movie name.")

if st.button("🧑‍🤝‍🧑 Show Cast and Crew"):
    if movie_input:
        movie = fetch_movie_details(movie_input)
        if movie:
            st.subheader("👥 Cast Members")
            cast = movie['credits']['cast'][:5]
            cols = st.columns(len(cast))
            for i, actor in enumerate(cast):
                with cols[i]:
                    if actor.get("profile_path"):
                        st.image(f"https://image.tmdb.org/t/p/w200{actor['profile_path']}", width=120)
                    st.markdown(f"**{actor['name']}**")
                    st.caption(f"_as {actor['character']}_")

            st.subheader("🛠️ Crew Members")
            crew = movie['credits']['crew'][:3]
            crew_cols = st.columns(len(crew))
            for i, member in enumerate(crew):
                with crew_cols[i]:
                    if member.get("profile_path"):
                        st.image(f"https://image.tmdb.org/t/p/w200{member['profile_path']}", width=120)
                    st.markdown(f"**{member['name']}**")
                    st.caption(f"_{member['job']}_")
        else:
            st.error("❌ Could not fetch movie.")

from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

if st.button("📊 Review Sentiment Analysis"):
    if movie_input:
        movie = fetch_movie_details(movie_input)
        if movie:
            reviews = movie.get("reviews", {}).get("results", [])
            if reviews:
                sentiments = {"positive": 0, "negative": 0, "neutral": 0}
                all_text = ""
                for review in reviews:
                    text = review['content']
                    analysis = TextBlob(text)
                    all_text += text + " "
                    polarity = analysis.sentiment.polarity
                    if polarity > 0:
                        sentiments["positive"] += 1
                    elif polarity < 0:
                        sentiments["negative"] += 1
                    else:
                        sentiments["neutral"] += 1

                st.subheader("📊 Sentiment Distribution")
                fig1, ax1 = plt.subplots(figsize=(3, 3))  # Smaller pie chart
                ax1.pie(
                    sentiments.values(),
                    labels=sentiments.keys(),
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={'fontsize': 8}  # Smaller text
                )
                ax1.axis("equal")
                st.pyplot(fig1)

                st.subheader("☁️ Word Cloud from Reviews")
                wc = WordCloud(width=400, height=200, background_color='white').generate(all_text)  # Smaller wordcloud
                fig2, ax2 = plt.subplots(figsize=(6, 3))  # Reduced size
                ax2.imshow(wc, interpolation='bilinear')
                ax2.axis("off")
                st.pyplot(fig2)
            else:
                st.warning("No reviews available.")
        else:
            st.error("❌ Could not fetch movie.")


if st.button("🎞️ Show Trailer"):
    if movie_input:
        movie = fetch_movie_details(movie_input)
        if movie:
            videos = movie.get("videos", {}).get("results", [])
            trailer = next((v for v in videos if v['site'] == 'YouTube' and ('trailer' in v['type'].lower() or 'teaser' in v['type'].lower())), None)
            if trailer:
                st.video(f"https://www.youtube.com/watch?v={trailer['key']}")
            else:
                st.warning("No trailer found.")
        else:
            st.error("❌ Could not fetch movie.")

if st.button("📈 Show Popularity & Runtime"):
    if movie_input:
        movie = fetch_movie_details(movie_input)
        if movie:
            popularity = movie.get("popularity", 0)
            runtime = movie.get("runtime", 0)
            df_chart = pd.DataFrame({
                "Metric": ["Popularity", "Runtime (min)"],
                "Value": [popularity, runtime]
            })
            st.subheader("📊 Popularity vs Runtime")
            st.bar_chart(df_chart.set_index("Metric"))
        else:
            st.error("❌ Could not fetch movie.")

if st.button("🎭 Show Movie Genres"):
    if movie_input:
        movie = fetch_movie_details(movie_input)
        if movie:
            genres = [genre['name'] for genre in movie.get("genres", [])]
            if genres:
                st.subheader("🎨 Top Genres:")
                for genre in genres:
                    st.markdown(f"- {genre}")
            else:
                st.info("No genre data available.")
        else:
            st.error("❌ Could not fetch movie.")
