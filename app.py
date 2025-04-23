# app.py (Streamlit app using FastAI model)
import streamlit as st
st.set_page_config(page_title="🎬 Anime Recommender", layout="wide")

import pandas as pd
from fastai.learner import load_learner
from recommend_fastai import recommend_anime_fastai, dls

@st.cache_resource
def load_model():
    return recommend_anime_fastai, dls

recommend_anime_fastai, dls = load_model()

# Theme toggle (moved outside sidebar for faster responsiveness)
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
            .stTextInput > div > input {
                background-color: #333; color: white;
            }
        </style>
    """, unsafe_allow_html=True)

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
    user_id = int(user_id_input)
    with st.spinner("🔎 Searching for the best anime for you..."):
        recs = recommend_anime_fastai(user_id, top_n, min_score)

    if recs.empty:
        st.warning("⚠️ No recommendations found. Showing popular anime instead.")
        recs = pd.read_csv("anime-dataset-2023.csv")
        recs = recs[['Name', 'Score', 'Genres', 'Type', 'Episodes']].sort_values(by='Score', ascending=False).head(top_n)
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
