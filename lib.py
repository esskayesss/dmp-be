import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import gears
import math

def preprocess_four(df):
    preprocessed_df = df[gears.EXPECTED_COLUMNS_FOUR].copy()

    ohe = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        feature_name_combiner=lambda x, y: f"shape_{y.replace(' ', '_').lower()}"
    )
    
    ohe.fit(pd.DataFrame({"shape": gears.shape_categories}))
    shape_encoded = ohe.transform(preprocessed_df[["shape"]])
    shape_columns = [f"shape_{cat.replace(' ', '_')}" for cat in ohe.categories_[0]]
    shape_df = pd.DataFrame(shape_encoded, columns=shape_columns, index=df.index)
    preprocessed_df = pd.concat([preprocessed_df, shape_df], axis=1)

    preprocessed_df['shape_encoded'] = preprocessed_df['shape'].map(gears.SHAPE_MAP)
    preprocessed_df['color_encoded'] = 22 + ord('D') - preprocessed_df['color'].str.upper().str[0].apply(ord)
    preprocessed_df['clarity_encoded'] = preprocessed_df['clarity'].map(gears.CLARITY_MAP)

    preprocessed_df['log_carat'] = np.log1p(preprocessed_df['carat'].clip(lower=0))
    preprocessed_df['carat_clarity'] = preprocessed_df['carat'] * preprocessed_df['clarity_encoded']
    preprocessed_df['carat_color'] = preprocessed_df['carat'] * preprocessed_df['color_encoded']
    preprocessed_df['clarity_color'] = preprocessed_df['clarity_encoded'] * preprocessed_df['color_encoded']

    scale_cols = [
        'carat', 'log_carat', 
        'carat_clarity', 'carat_color', 'clarity_color', 
    ]
    preprocessed_df[scale_cols] = gears.numerical_scaler_fourparams.transform(preprocessed_df[scale_cols])
    return preprocessed_df[gears.fourparams]


def preprocess_all(df):
    preprocessed_df = df[gears.EXPECTED_COLUMNS_ALL].copy()

    ohe = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        feature_name_combiner=lambda x, y: f"shape_{y.replace(' ', '_').lower()}"
    )
    
    ohe.fit(pd.DataFrame({"shape": gears.shape_categories}))
    shape_encoded = ohe.transform(preprocessed_df[["shape"]])
    shape_columns = [f"shape_{cat.replace(' ', '_')}" for cat in ohe.categories_[0]]
    shape_df = pd.DataFrame(shape_encoded, columns=shape_columns, index=df.index)
    preprocessed_df = pd.concat([preprocessed_df, shape_df], axis=1)

    preprocessed_df['shape_encoded'] = preprocessed_df['shape'].map(gears.SHAPE_MAP)
    preprocessed_df['color_encoded'] = 22 + ord('D') - preprocessed_df['color'].str.upper().str[0].apply(ord)
    preprocessed_df['cut_encoded'] = preprocessed_df['cut'].map(gears.CUT_MAP)
    preprocessed_df['fluorescence_encoded'] = preprocessed_df['fluorescence'].map(gears.FLUORESCENCE_MAP)
    preprocessed_df['clarity_encoded'] = preprocessed_df['clarity'].map(gears.CLARITY_MAP)

    preprocessed_df['volume'] = preprocessed_df['x'] * preprocessed_df['y'] * preprocessed_df['z']
    preprocessed_df['log_carat'] = np.log1p(preprocessed_df['carat'].clip(lower=0))
    preprocessed_df['carat_clarity'] = preprocessed_df['carat'] * preprocessed_df['clarity_encoded']
    preprocessed_df['carat_color'] = preprocessed_df['carat'] * preprocessed_df['color_encoded']
    preprocessed_df['carat_cut'] = preprocessed_df['carat'] * preprocessed_df['cut_encoded']
    preprocessed_df['carat_fluorescence'] = preprocessed_df['carat'] * preprocessed_df['fluorescence_encoded']
    preprocessed_df['carat_volume'] = preprocessed_df['carat'] * preprocessed_df['volume']
    preprocessed_df['carat_square'] = preprocessed_df['carat'] * preprocessed_df['x'] * preprocessed_df['y']
    preprocessed_df['clarity_color'] = preprocessed_df['clarity_encoded'] * preprocessed_df['color_encoded']
    preprocessed_df['depth_to_table'] = preprocessed_df['depth'] / preprocessed_df['table']
    preprocessed_df['volume_to_carat'] = preprocessed_df['volume'] / preprocessed_df['carat']
    preprocessed_df['carat_to_volume'] = preprocessed_df['carat'] / preprocessed_df['volume']
    preprocessed_df['symmetry_xy'] = (preprocessed_df['x'] - preprocessed_df['y']).abs()

    preprocessed_df['depth_to_table'] = (
        preprocessed_df['depth_to_table'].replace([np.inf, -np.inf], np.nan)
        .fillna(preprocessed_df['depth_to_table'].median())
    )

    preprocessed_df['volume_to_carat'] = (
        preprocessed_df['volume_to_carat'].replace([np.inf, -np.inf], np.nan)
        .fillna(preprocessed_df['volume_to_carat'].median())
    )

    preprocessed_df['carat_to_volume'] = (
        preprocessed_df['carat_to_volume'].replace([np.inf, -np.inf], np.nan)
        .fillna(preprocessed_df['carat_to_volume'].median())
    )

    scale_cols = [
        'carat', 
        'x', 'y', 'z', 
        'depth', 'table', 
        'volume', 'log_carat', 
        'carat_clarity', 'carat_color', 
        'carat_cut', 'carat_fluorescence', 
        'carat_volume', 'carat_square', 
        'clarity_color', 'depth_to_table', 
        'volume_to_carat', 'carat_to_volume', 
    ]
    preprocessed_df[scale_cols] = gears.numerical_scaler_allparams.transform(preprocessed_df[scale_cols])
    return preprocessed_df[gears.allparams]


def price_from_pred(pred: np.ndarray) -> int:
    y_true_orig = gears.price_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
    pred_price = math.ceil(np.expm1(y_true_orig))
    return pred_price
