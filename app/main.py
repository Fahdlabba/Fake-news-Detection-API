from fastapi import FastAPI
from pydantic import BaseModel
from model.model import prediction

app = FastAPI()


class TextIn(BaseModel):
    inputText: str


class PredictionOut(BaseModel):
    Prediction: str


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict", response_model=PredictionOut)
async def predict(payload: TextIn):
    text2 = payload.inputText
    classP=prediction(text2)
    return {"Prediction": classP}