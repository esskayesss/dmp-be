import joblib
from xgboost import DMatrix
import xgboost as xgb
from catboost import CatBoostRegressor
import gears
import lib
import pandas as pd


xgb_allparams = joblib.load(f'./data/allparams/{gears.XGBOOST_PATH}')
xgb_finetuned_allparams = xgb.Booster()
xgb_finetuned_allparams.load_model(f'./data/allparams/{gears.FINE_XGBOOST_PATH}')
xgb_fourparams = joblib.load(f'./data/fourparams/{gears.XGBOOST_PATH}')
xgb_finetuned_fourparams = xgb.Booster()
xgb_finetuned_fourparams.load_model(f'./data/fourparams/{gears.FINE_XGBOOST_PATH}')

models_allparams = {
    'ExtraTrees': joblib.load(f'./data/allparams/{gears.EXTRATREES_PATH}'),
    'RandomForest': joblib.load(f'./data/allparams/{gears.RANDOMFOREST_PATH}'),
    'LightGBM': joblib.load(f'./data/allparams/{gears.LIGHTGBM_PATH}'),
    'CatBoost': CatBoostRegressor().load_model(f'./data/allparams/{gears.CATBOOST_PATH}'),
}

models_fourparams = {
    'ExtraTrees': joblib.load(f'./data/fourparams/{gears.EXTRATREES_PATH}'),
    'RandomForest': joblib.load(f'./data/fourparams/{gears.RANDOMFOREST_PATH}'),
    'LightGBM': joblib.load(f'./data/fourparams/{gears.LIGHTGBM_PATH}'),
    'CatBoost': CatBoostRegressor().load_model(f'./data/fourparams/{gears.CATBOOST_PATH}'),
}

def ensemble_allparams(preprocessed: pd.DataFrame) -> dict:
    # ensure gears.allparams are in preprocessed
    if not all(col in preprocessed.columns for col in gears.allparams):
        raise ValueError("preprocessed must contain all columns in gears.allparams")
    results: dict[str, int] = {}
    for model_name, model in models_allparams.items():
        pred = model.predict(preprocessed)
        price = lib.price_from_pred(pred)
        results[model_name] = price
    
    preprocessed_dmatrix = DMatrix(preprocessed)
    pred = xgb_allparams.predict(preprocessed_dmatrix)
    price = lib.price_from_pred(pred)
    results['XGBoost'] = price
    
    pred = xgb_finetuned_allparams.predict(preprocessed_dmatrix)
    price = lib.price_from_pred(pred)
    results['XGBoost_finetuned'] = price
    return results


def ensemble_fourparams(preprocessed: pd.DataFrame) -> dict:
    # ensure gears.fourparams are in preprocessed
    if not all(col in preprocessed.columns for col in gears.fourparams):
        raise ValueError("preprocessed must contain all columns in gears.fourparams")
    results: dict[str, int] = {}
    for model_name, model in models_fourparams.items():
        pred = model.predict(preprocessed)
        price = lib.price_from_pred(pred)
        results[model_name] = price
    
    preprocessed_dmatrix = DMatrix(preprocessed)
    pred = xgb_fourparams.predict(preprocessed_dmatrix)
    price = lib.price_from_pred(pred)
    results['XGBoost'] = price
    
    pred = xgb_finetuned_fourparams.predict(preprocessed_dmatrix)
    price = lib.price_from_pred(pred)
    results['XGBoost_finetuned'] = price
    return results