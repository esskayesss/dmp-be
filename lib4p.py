import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

# Mappings for categorical features
cut_mapping = {
    "Good": 2,
    "Very Good": 3,
    "Excellent": 4,
    "Ideal": 5,
    "Astor": 6,
    "True Hearts": 7,
}
color_mapping = {
    "M": 0,
    "L": 1,
    "K": 2,
    "J": 3,
    "I": 4,
    "H": 5,
    "G": 6,
    "F": 7,
    "E": 8,
    "D": 9,
}
clarity_mapping = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
    "FL": 8,
}
fluorescence_mapping = {
    "Strong": 0,
    "Strong Blue": 0,
    "Medium": 1,
    "Medium Blue": 1,
    "Slight": 2,
    "Faint": 2,
    "Negligible": 3,
    "None": 3,
}

with open("./data/fourp/scaler_4p.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("./data/fourp/feature_cols_reduced_4p.pkl", "rb") as f:
    feature_cols_reduced = pickle.load(f)
with open("./data/fourp/feature_weights_4p.pkl", "rb") as f:
    feature_weights = pickle.load(f)
with open("./data/fourp/feature_cols_reduced_4p.pkl", "rb") as f:
    feature_cols_reduced = pickle.load(f)
feature_cols_reduced = [col.replace(" ", "_") for col in feature_cols_reduced]
print("Corrected feature_cols_reduced:", feature_cols_reduced)
with open("./data/fourp/feature_cols_reduced_4p.pkl", "wb") as f:
    pickle.dump(feature_cols_reduced, f)


def preprocess_user_query(user_df):
    try:
        df = user_df.copy()
        df["color"] = df["color"].map(color_mapping)
        df["clarity"] = df["clarity"].map(clarity_mapping)

        # Validate that all mappings succeeded (i.e., no NaN values after mapping)
        for col in ["color", "clarity"]:
            if df[col].isna().any():
                raise ValueError(
                    f"Invalid value for {col}: {df[col].iloc[0]} not in mapping"
                )

        # Apply one-hot encoding for shape, using the same categories as training
        ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            feature_name_combiner=lambda x, y: f"shape_{y.replace(' ', '_')}",
        )
        shape_categories = [
            "asscher",
            "cushion",
            "cushion_modified",
            "emerald",
            "heart",
            "marquise",
            "oval",
            "pear",
            "princess",
            "radiant",
            "round",
            "square_radiant",
        ]
        ohe.fit(pd.DataFrame({"shape": shape_categories}))
        shape_encoded = ohe.transform(df[["shape"]])
        shape_columns = [f"shape_{cat.replace(' ', '_')}" for cat in ohe.categories_[0]]
        shape_df = pd.DataFrame(shape_encoded, columns=shape_columns, index=df.index)
        df = df.drop(columns=["shape"])
        df = pd.concat([df, shape_df], axis=1)

        # Handle missing values for robustness
        for col in ["color", "clarity"]:
            df[col] = df[col].fillna(df[col].mode()[0])
        numerical_cols = ["carat"]
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())

        # Feature engineering (in original units)
        df["log_carat"] = np.log1p(df["carat"].clip(lower=0))
        df["carat_clarity"] = df["carat"] * df["clarity"]
        df["carat_color"] = df["carat"] * df["color"]
        df["clarity_color"] = df["clarity"] * df["color"]
        df["carat_squared"] = df["carat"] ** 2
        scale_cols = [
            "carat",
            "log_carat",
            "carat_clarity",
            "carat_color",
            "clarity_color",
            "carat_squared",
        ]
        df[scale_cols] = scaler.transform(df[scale_cols])

        # Select the reduced feature columns
        user_features = df[feature_cols_reduced].values.astype(np.float32)

        # Apply feature importance weights
        weights = np.array([feature_weights[col] for col in feature_cols_reduced])
        user_features = user_features * weights

        return user_features

    except Exception as e:
        raise RuntimeError(f"Error preprocessing user query: {str(e)}")
