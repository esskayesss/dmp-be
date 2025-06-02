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

def knn_predict_allparams(preprocessed: pd.DataFrame, k: int = 12) -> float:
    # ensure gears.allparams are in preprocessed
    if not all(col in preprocessed.columns for col in gears.allparams):
        raise ValueError("preprocessed must contain all columns in gears.allparams")
    
    indices, _ = index_allparams.knn_query(preprocessed.values, k)
    predicted = np.median(y[indices])
    return lib.price_from_pred(predicted)


def knn_predict_fourparams(preprocessed: pd.DataFrame, k: int = 12) -> float:
    # ensure gears.fourparams are in preprocessed
    if not all(col in preprocessed.columns for col in gears.fourparams):
        raise ValueError("preprocessed must contain all columns in gears.fourparams")
    
    indices, _ = index_4p.knn_query(preprocessed.values, k)
    predicted = np.median(y[indices])
    return lib.price_from_pred(predicted)