# app.py (Streamlit app using FastAI model)
import streamlit as st
st.set_page_config(page_title="🎬 Anime Recommender", layout="wide")

import pandas as pd
import os
import gdown
from fastai.learner import load_learner

MODEL_URL = "https://drive.google.com/uc?id=1mchVb5zKND4Da0WjKeOsrYpesGRE4Gyt"
MODEL_PATH = "anime_recommender_fastai.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📦 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

learn = load_learner(MODEL_PATH)
dls = learn.dls

def recommend_anime_fastai(user_id, top_n=10, min_score=7.0):
    try:
        if user_id not in dls.classes['user_id']:
            return pd.DataFrame(columns=["Name", "Score", "Genres", "Type", "Episodes", "pred_rating"])

        max_valid_index = learn.model.i_weight.num_embeddings
        valid_anime_ids = dls.classes['anime_id'][:max_valid_index]

        df = pd.DataFrame({
            'user_id': [user_id] * len(valid_anime_ids),
            'anime_id': valid_anime_ids
        })

        df['user_id'] = pd.Categorical(df['user_id'], categories=dls.classes['user_id'])
        df['anime_id'] = pd.Categorical(df['anime_id'], categories=dls.classes['anime_id'])

        df = df[df['user_id'].notna() & df['anime_id'].notna()]
        df_encoded = df.copy()
        df_encoded['user_id'] = df['user_id'].cat.codes
        df_encoded['anime_id'] = df['anime_id'].cat.codes

        test_dl = dls.test_dl(df_encoded)
        preds, _ = learn.get_preds(dl=test_dl)
        df['pred_rating'] = preds.numpy()

        anime_df = pd.read_csv("anime-dataset-2023.csv")
        anime_df['Score'] = pd.to_numeric(anime_df['Score'], errors='coerce')
        anime_df = anime_df[['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes']]
        merged = df.merge(anime_df, on='anime_id', how='left')
        merged = merged[merged['Score'] >= min_score]

        return merged.sort_values('pred_rating', ascending=False).head(top_n)
    except Exception as e:
        st.error(f"🚫 Internal error: {e}")
        return pd.DataFrame(columns=["Name", "Score", "Genres", "Type", "Episodes", "pred_rating"])

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
