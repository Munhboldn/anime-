import pandas as pd
from fastai.learner import load_learner
from pathlib import Path
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load trained model
def load_model(model_path="anime_recommender_fastai.pkl"):
    try:
        model_path = str(Path(model_path))  # Avoid WindowsPath issues
        learn = load_learner(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return learn
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}\n{traceback.format_exc()}")
        raise

# Recommend anime for a user
def recommend_anime_fastai(user_id, top_n=10, min_score=7.0, anime_df=None):
    try:
        learn = load_model()  # Load model if not passed
        dls = learn.dls

        if user_id not in dls.classes['user_id']:
            logger.warning(f"User ID {user_id} not found in model classes")
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

        if anime_df is None:
            logger.warning("No anime_df provided, loading default dataset")
            anime_df = pd.read_csv("anime-dataset-2023.csv")
            anime_df['Score'] = pd.to_numeric(anime_df['Score'], errors='coerce')
            anime_df = anime_df[['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes']]

        merged = df.merge(anime_df, on='anime_id', how='left')
        merged = merged[pd.to_numeric(merged['Score'], errors='coerce') >= min_score]

        recommendations = merged.sort_values('pred_rating', ascending=False).head(top_n)
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}\n{traceback.format_exc()}")
        raise
