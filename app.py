import streamlit as st
st.set_page_config(page_title="🎬 Anime Recommender", layout="wide")

import pandas as pd
import os
import urllib.request
from pathlib import Path
import logging
from recommend_fastai import load_model, recommend_anime_fastai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model and dataset paths
MODEL_URL = "https://www.dropbox.com/scl/fi/ucp9m89b244cmsp61cax4/anime_recommender_fastai.pkl?rlkey=lt2awixz2e60wgyngh318h6rt&st=vlcosxta&raw=1"
MODEL_PATH = "anime_recommender_fastai.pkl"
DATASET_PATH = "anime-dataset-2023.csv"

# Load dataset
try:
    anime_df = pd.read_csv(DATASET_PATH)
    anime_df = anime_df[['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes']]
    anime_df['Score'] = pd.to_numeric(anime_df['Score'], errors='coerce')
    logger.info("Successfully loaded anime dataset")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    st.error(f"🚫 Failed to load anime dataset: {e}")
    st.stop()

# Download model if not exists
try:
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info(f"Downloaded model to {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to download model: {e}")
    st.error(f"🚫 Failed to download model: {e}")
    st.stop()

# Load model
try:
    learn = load_model(MODEL_PATH)  # Uses str(Path(MODEL_PATH)) internally
    dls = learn.dls
    logger.info("Successfully loaded model")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    st.error(f"🚫 Failed to load model: {e}")
    st.stop()

# Theme toggle
mode = st.sidebar.radio("🌗 Theme Mode", ["Light", "Dark"], key="theme_mode")
if mode == "Dark":
    st.markdown("""
        <style>
            body, .main, .block-container {
                background-color: #1e1e1e;
                color: white;
            }
            .stButton > button {
                background-color: #4CAF50; color: white;
            }
            .stTextInput > div > input, .stSelectbox > div > select {
                background-color: #333; color: white;
            }
        </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("🎌 Welcome to Your Anime Recommendation Portal")
st.markdown("Easily find anime you might love — based on your past ratings")

with st.sidebar:
    st.header("⚙️ Settings")
    valid_user_ids = dls.classes['user_id']
    user_id_input = st.selectbox("👤 Choose your User ID", valid_user_ids, key="user_id")
    top_n = st.slider("📋 Number of recommendations", 5, 20, 10, key="top_n")
    min_score = st.slider("🌟 Minimum anime rating (filter)", 1.0, 10.0, 7.0, step=0.5, key="min_score")
    anime_search = st.text_input("🔎 Search Anime Name", key="anime_search")
    submit = st.button("📥 Recommend", key="submit_btn")

if submit:
    try:
        user_id = user_id_input  # Keep as string to avoid int conversion issues
        with st.spinner("🔎 Searching for the best anime for you..."):
            recs = recommend_anime_fastai(user_id, top_n, min_score, anime_df=anime_df)

        if recs.empty:
            st.warning("⚠️ No recommendations found. Showing popular anime instead.")
            recs = anime_df.sort_values(by='Score', ascending=False).head(top_n)
            recs['pred_rating'] = recs['Score']
        else:
            st.success(f"✅ Top {len(recs)} personalized picks for User ID {user_id}:")

        if anime_search:
            recs = recs[recs['Name'].str.contains(anime_search, case=False, na=False)]
            st.info(f"🔍 Found {len(recs)} matches for '{anime_search}'")

        # Expandable anime cards
        for i, row in recs.iterrows():
            with st.expander(f"{i+1}. {row['Name']} ({row['pred_rating']:.2f}/10)", expanded=False):
                st.write(f"⭐ **Predicted Rating:** {row['pred_rating']:.2f}/10")
                st.write(f"🎯 **MAL Score:** {row['Score']}")
                st.write(f"🎭 **Genres:** {row['Genres']}")
                st.write(f"📺 **Type:** {row['Type']} | 📦 **Episodes:** {row['Episodes']}")
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        st.error(f"🚫 Error generating recommendations: {e}")
