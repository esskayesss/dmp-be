import pandas as pd
import numpy as np
import hnswlib
import gears
import lib

index_allparams = hnswlib.Index(space="l2", dim=len(gears.allparams))
index_allparams.load_index("./data/knn/hnswlib_index_l2_allparams.bin")

index_4p = hnswlib.Index(space="l2", dim=len(gears.fourparams))
index_4p.load_index("./data/knn/hnswlib_index_l2_fourparams.bin")

y = np.load("./data/knn/y.npy")


def indices_to_predictions(indices: np.ndarray, distances: np.ndarray) -> [int, int, int]:
    k_pred = np.median(y[indices[0]])
    k_pred = lib.price_from_pred(k_pred)

    dists = distances[0]
    sorted_indices = np.argsort(dists)
    sorted_dists = dists[sorted_indices]

    gaps = np.diff(sorted_dists)
    elbow_idx = np.argmax(gaps) + 1
    valid_indices = indices[0][sorted_indices[:elbow_idx]]
    if len(valid_indices) == 0:
        pred_price = k_pred
    else:
        pred_price = lib.price_from_pred(np.median(y[valid_indices]))

    return [pred_price, k_pred, len(valid_indices)]


def knn_predict_allparams(preprocessed: pd.DataFrame, k: int = 12) -> [int, int]:
    # ensure gears.allparams are in preprocessed
    if not all(col in preprocessed.columns for col in gears.allparams):
        raise ValueError("preprocessed must contain all columns in gears.allparams")
    
    indices, distances = index_allparams.knn_query(preprocessed.values, k)
    return indices_to_predictions(indices, distances)


def knn_predict_fourparams(preprocessed: pd.DataFrame, k: int = 12) -> [int, int]:
    # ensure gears.fourparams are in preprocessed
    if not all(col in preprocessed.columns for col in gears.fourparams):
        raise ValueError("preprocessed must contain all columns in gears.fourparams")
    
    indices, distances = index_4p.knn_query(preprocessed.values, k)
    return indices_to_predictions(indices, distances)
