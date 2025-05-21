import numpy as np
import hnswlib
import pickle

with open("./data/fourp/feature_cols_reduced_4p.pkl", "rb") as f:
    feature_cols_reduced = pickle.load(f)
feature_cols_reduced = [col.replace(" ", "_") for col in feature_cols_reduced]
print("Corrected feature_cols_reduced:", feature_cols_reduced)
with open("./data/fourp/feature_cols_reduced_4p.pkl", "wb") as f:
    pickle.dump(feature_cols_reduced, f)

dim = len(feature_cols_reduced)
index = hnswlib.Index(space="l2", dim=dim)
index.load_index("./data/fourp/hnswlib_index_l1_weighted_4p.bin")
y = np.load("./data/fourp/y_4p.npy")
with open("./data/fourp/log_price_metadata_4p.pkl", "rb") as f:
    log_price_metadata = pickle.load(f)
log_price_mean = log_price_metadata["mean"]
log_price_std = log_price_metadata["std"]


def knn_predict(user_features, k=12) -> float:
    indices, _ = index.knn_query(user_features, k=k)
    pred_log_price = np.median(y[indices[0]])
    pred_price = np.expm1(pred_log_price * log_price_std + log_price_mean)
    return pred_price
