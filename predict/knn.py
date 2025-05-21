import pandas as pd
import numpy as np
import hnswlib
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


with open("./data/allparams/feature_cols_reduced.pkl", "rb") as f:
    feature_cols_reduced = pickle.load(f)
feature_cols_reduced = [col.replace(" ", "_") for col in feature_cols_reduced]
print("Corrected feature_cols_reduced:", feature_cols_reduced)
with open("./data/allparams/feature_cols_reduced.pkl", "wb") as f:
    pickle.dump(feature_cols_reduced, f)

# Load the HNSW index
dim = len(feature_cols_reduced)  # 20 features
index = hnswlib.Index(space="l2", dim=dim)
index.load_index("./data/allparams/hnswlib_index_l1_weighted.bin")

# Load the target values (y)
y = np.load("./data/allparams/y.npy")

# Load log_price_metadata
with open("./data/allparams/log_price_metadata.pkl", "rb") as f:
    log_price_metadata = pickle.load(f)
log_price_mean = log_price_metadata["mean"]
log_price_std = log_price_metadata["std"]


def knn_predict(user_features, k=12) -> float:
    indices, _ = index.knn_query(user_features, k=k)
    pred_log_price = np.median(y[indices[0]])
    pred_price = np.expm1(pred_log_price * log_price_std + log_price_mean)
    return pred_price
