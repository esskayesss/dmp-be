from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from predict.index import DiamondIndex, DiamondIndex4P
from diamond import Diamond, Diamond4P
import time


class Diamond4PBody(BaseModel):
    carat: float
    color: str
    clarity: str
    shape: str


class DiamondBody(Diamond4PBody):
    cut: str
    depth: float
    table: float
    x: float
    y: float
    z: float
    fluorescence: str


SHAPE_ADJUSTMENTS = {
    "round": 0.0,  # Baseline
    "asscher": 0.003,  # +0.3%
    "cushion": 0.002,  # +0.2%
    "cushion_modified": 0.0015,  # +0.15%
    "emerald": 0.001,  # +0.1%
    "heart": -0.001,  # -0.1%
    "marquise": -0.0015,  # -0.15%
    "oval": 0.0005,  # +0.05%
    "pear": -0.0005,  # -0.05%
    "princess": 0.0025,  # +0.25%
    "radiant": 0.002,  # +0.2%
    "square_radiant": 0.001,  # +0.1%
}

index = DiamondIndex()
index4p = DiamondIndex4P()
app = FastAPI()
# app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.post("/predict")
async def predict(data: DiamondBody):
    start_time = time.time()
    try:
        diamond = Diamond(
            carat=data.carat,
            cut=data.cut,
            color=data.color,
            clarity=data.clarity,
            depth=data.depth,
            table=data.table,
            x=data.x,
            y=data.y,
            z=data.z,
            shape=data.shape,
            fluorescence=data.fluorescence,
        )
        predicted_price = index.get(diamond)
        # Apply shape-based adjustment
        adjustment_factor = 1 + SHAPE_ADJUSTMENTS.get(data.shape.lower(), 0.0)
        adjusted_price = predicted_price * adjustment_factor
        # Round to nearest integer
        final_price = round(adjusted_price)
        return {
            "predicted_price": final_price,
            "processing_time": time.time() - start_time,
        }
    except ValueError as e:
        return {"error": str(e)}


@app.post("/predict4p")
async def predict(data: Diamond4PBody):
    start_time = time.time()
    try:
        diamond = Diamond4P(
            carat=data.carat,
            color=data.color,
            clarity=data.clarity,
            shape=data.shape,
        )
        predicted_price = index4p.get(diamond)
        # Apply shape-based adjustment
        adjustment_factor = 1 + SHAPE_ADJUSTMENTS.get(data.shape.lower(), 0.0)
        adjusted_price = predicted_price * adjustment_factor
        # Round to nearest integer
        final_price = round(adjusted_price)
        return {
            "predicted_price": final_price,
            "processing_time": time.time() - start_time,
        }
    except ValueError as e:
        return {"error": str(e)}


@app.post("/predict-k")
async def predictk(data: DiamondBody):
    start_time = time.time()
    try:
        diamond = Diamond(
            carat=data.carat,
            cut=data.cut,
            color=data.color,
            clarity=data.clarity,
            depth=data.depth,
            table=data.table,
            x=data.x,
            y=data.y,
            z=data.z,
            shape=data.shape,
            fluorescence=data.fluorescence,
        )
        predictions = {}
        for k in range(1, 50):
            predictions[k] = diamond.predict_price(k=k)
        return {
            "predicted_price": predictions,
            "processing_time": time.time() - start_time,
        }
    except ValueError as e:
        return {"error": str(e)}


@app.post("/predict4p-k")
async def predict4pk(data: Diamond4PBody):
    start_time = time.time()
    try:
        diamond = Diamond4P(
            carat=data.carat,
            color=data.color,
            clarity=data.clarity,
            shape=data.shape,
        )
        predictions = {}
        for k in range(1, 50):
            predictions[k] = diamond.predict_price(k=k)
        return {
            "predicted_price": predictions,
            "processing_time": time.time() - start_time,
        }
    except ValueError as e:
        return {"error": str(e)}
