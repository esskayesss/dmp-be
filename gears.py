import joblib

DATA_DIR = './data'
numerical_scaler_allparams = joblib.load(f"{DATA_DIR}/numerical_scaler_allparams.pkl")
numerical_scaler_fourparams = joblib.load(f"{DATA_DIR}/numerical_scaler_fourparams.pkl")
price_scaler = joblib.load(f"{DATA_DIR}/price_scaler.pkl")

EXTRATREES_PATH = 'extratrees_best.pkl'
RANDOMFOREST_PATH = 'randomforest_best.pkl'
LIGHTGBM_PATH = 'lightgbm_best.pkl'
XGBOOST_PATH = 'xgboost_best.pkl'
FINE_XGBOOST_PATH = 'xgboost_finetuned.json'
CATBOOST_PATH = 'catboost_best.cbm'

allparams = ['carat', 'x', 'y', 'z', 'depth', 'table', 'shape_asscher',
    'shape_cushion', 'shape_cushion_modified', 'shape_emerald',
    'shape_heart', 'shape_marquise', 'shape_oval', 'shape_pear',
    'shape_princess', 'shape_radiant', 'shape_round',
    'shape_square_radiant', 'shape_encoded', 'color_encoded', 'cut_encoded',
    'fluorescence_encoded', 'clarity_encoded', 'volume', 'log_carat',
    'carat_clarity', 'carat_color', 'carat_cut', 'carat_fluorescence',
    'carat_volume', 'carat_square', 'clarity_color', 'depth_to_table',
    'volume_to_carat', 'carat_to_volume', 'symmetry_xy']

fourparams = ['carat',
    'shape_asscher', 'shape_cushion', 'shape_cushion_modified', 'shape_emerald',
    'shape_heart', 'shape_marquise', 'shape_oval', 'shape_pear',
    'shape_princess', 'shape_radiant', 'shape_round',
    'shape_square_radiant', 
    'shape_encoded', 
    'color_encoded',
    'clarity_encoded', 
    'log_carat', 'carat_clarity', 'carat_color', 'clarity_color']

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

EXPECTED_COLUMNS_ALL = [
    'carat',
    'shape',
    'cut',
    'color',
    'clarity',
    'fluorescence',
    'x',
    'y',
    'z',
    'depth',
    'table'
]

EXPECTED_COLUMNS_FOUR = [
    'carat',
    'shape',
    'color',
    'clarity',
]

SHAPE_MAP = {
    'princess': 0,
    'pear': 1,
    'round': 2,
    'marquise': 3,
    'heart': 4,
    'emerald': 5,
    'oval': 6,
    'square radiant': 7,
    'cushion modified': 8,
    'asscher': 9,
    'radiant': 10,
    'cushion': 11
}

CUT_MAP = {
    'Astor': 0,
    'Excellent': 1,
    'Very Good': 2,
    'Ideal': 3,
    'Good': 4,
    'True Hearts': 5
}

FLUORESCENCE_MAP = {
    'Medium': 0,
    'Slight': 1,
    'Strong Blue': 2,
    'Faint': 3,
    'None': 4,
    'Strong': 5,
    'Negligible': 6,
    'Medium Blue': 7
}

CLARITY_MAP = {
    'I3': 0,
    'I2': 1,
    'I1': 2,
    'SI2': 3,
    'SI1': 4,
    'VS2': 5,
    'VS1': 6,
    'VVS2': 7,
    'VVS1': 8,
    'IF': 9,
    'FL': 10
}
