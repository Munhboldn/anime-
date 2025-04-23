# recommend_fastai.py
import pandas as pd
from fastai.learner import load_learner

# Load metadata
anime_df = pd.read_csv("anime-dataset-2023.csv")
anime_df = anime_df[['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes']]
anime_df['Score'] = pd.to_numeric(anime_df['Score'], errors='coerce')

# Load trained model
learn = load_learner("anime_recommender_fastai.pkl")
dls = learn.dls

# Recommend anime for a user
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

        merged = df.merge(anime_df, on='anime_id', how='left')
        merged = merged[pd.to_numeric(merged['Score'], errors='coerce') >= min_score]

        return merged.sort_values('pred_rating', ascending=False).head(top_n)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame(columns=["Name", "Score", "Genres", "Type", "Episodes", "pred_rating"])
