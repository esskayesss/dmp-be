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
        predicted_price: dict = index.get(diamond)
        return {
            "predictions": {**predicted_price},
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
        predicted_price: dict = index4p.get(diamond)
        return {
            "predictions": {**predicted_price},
            "processing_time": time.time() - start_time,
        }
    except ValueError as e:
        return {"error": str(e)}