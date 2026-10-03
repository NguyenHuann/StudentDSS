import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Xác định đường dẫn tương đối tới file CSV trong cùng thư mục
BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "recommendation_items.csv"

CONTENT_FEATURES = [
    "low_grade",
    "low_pass_rate",
    "failed_burden",
    "evaluation_issue",
    "workload_gap",
    "high_performance",
]

RAW_FEATURES = [
    "1st_sem_enrolled",
    "1st_sem_evaluations",
    "1st_sem_without_evaluations",
    "1st_sem_approved",
    "1st_sem_failed",
    "1st_sem_grade",
    "1st_sem_pass_rate",
]

MAX_GRADE = 20.0

def load_catalog():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy catalog: {CATALOG_PATH}")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    return catalog.reset_index(drop=True)

# Tải sẵn catalog vào bộ nhớ khi import module
CATALOG_DF = load_catalog()

def add_derived_features(student_df):
    df = student_df.copy()
    df["1st_sem_pass_rate"] = np.where(
        df["1st_sem_evaluations"] > 0,
        df["1st_sem_approved"] / df["1st_sem_evaluations"],
        0.0,
    )
    df["1st_sem_pass_rate"] = df["1st_sem_pass_rate"].clip(0, 1)
    return df

def create_student_content_profile(student_df):
    df = add_derived_features(student_df)
    grade_norm = (df["1st_sem_grade"] / MAX_GRADE).clip(0, 1)
    enrolled = df["1st_sem_enrolled"].replace(0, np.nan)

    semantic = pd.DataFrame(index=df.index)
    semantic["low_grade"] = (1.0 - grade_norm).clip(0, 1)
    semantic["low_pass_rate"] = (1.0 - df["1st_sem_pass_rate"]).clip(0, 1)
    semantic["failed_burden"] = (df["1st_sem_failed"] / enrolled).fillna(0).clip(0, 1)
    semantic["evaluation_issue"] = (df["1st_sem_without_evaluations"] / enrolled).fillna(0).clip(0, 1)
    semantic["workload_gap"] = (1.0 - (df["1st_sem_approved"] / enrolled).fillna(0)).clip(0, 1)
    semantic["high_performance"] = (grade_norm * df["1st_sem_pass_rate"]).clip(0, 1)
    return semantic[CONTENT_FEATURES]

def get_item_matrix(catalog):
    return catalog[CONTENT_FEATURES].to_numpy(dtype=float)

def explain_similarity(student_vector, item_vector):
    s = np.asarray(student_vector, dtype=float).reshape(-1)
    i = np.asarray(item_vector, dtype=float).reshape(-1)
    s_norm, i_norm = np.linalg.norm(s), np.linalg.norm(i)
    if s_norm == 0 or i_norm == 0:
        return pd.Series(0.0, index=CONTENT_FEATURES)
    contributions = (s * i) / (s_norm * i_norm)
    return pd.Series(contributions, index=CONTENT_FEATURES)

def recommend_study_plans(student_data, catalog=CATALOG_DF, top_n=3, low_confidence_threshold=None):
    student_profile = create_student_content_profile(student_data)
    student_vectors = student_profile.to_numpy(dtype=float)
    item_vectors = get_item_matrix(catalog)

    similarities = cosine_similarity(student_vectors, item_vectors)
    rows = []

    for student_idx in range(len(student_profile)):
        order = np.argsort(-similarities[student_idx], kind="stable")[:top_n]
        best_score = float(similarities[student_idx, order[0]])

        for rank, item_idx in enumerate(order, start=1):
            item = catalog.iloc[item_idx]
            contribution = explain_similarity(student_vectors[student_idx], item_vectors[item_idx])
            top_dimensions = contribution.sort_values(ascending=False).head(3)

            rows.append({
                "rank": rank,
                "item_id": item["item_id"],
                "item_name": item["item_name"],
                "category": item["category"],
                "description": item["description"],
                "action_plan": item["action_plan"],
                "similarity_score": float(similarities[student_idx, item_idx]),
                "low_confidence": (low_confidence_threshold is not None and best_score < low_confidence_threshold),
                "top_matching_dimensions": ", ".join(top_dimensions.index),
            })

    return pd.DataFrame(rows), student_profile

def recommend_one(student_dict, top_n=3):
    student_df = pd.DataFrame([student_dict])
    result, profile = recommend_study_plans(student_df, catalog=CATALOG_DF, top_n=top_n)
    return result, profile.iloc[0]