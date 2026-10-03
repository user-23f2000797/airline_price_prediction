"""
FastAPI application for predicting airline prices
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
import os
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
preprocessor_path = os.path.join(os.path.dirname(__file__), "preprocess.pkl")
si_path = os.path.join(os.path.dirname(__file__), "si.pkl")
si_cat_path = os.path.join(os.path.dirname(__file__), "si_cat.pkl")


try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(preprocessor_path)
    si = joblib.load(si_path)
    si_cat = joblib.load(si_cat_path)

except FileNotFoundError:
    raise FileNotFoundError(f"file not found. Please ensure the model is trained and saved.")



app = FastAPI(
    title="Airline Price Prediction API",
    version="1.0",
    description="API for predicting airline prices based on input features."
)

class FlightData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    airline: str
    source: str
    destination: str
    departure: str
    stops: str
    arrival: str
    class_: str = Field(alias="class")
    duration: float
    days_left: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Airline Price Prediction API. Use the /predict endpoint to get predictions."}


@app.post("/predict")
def predict_price(data: FlightData):
    """
    Predict airline price based on input features.
    """
    try:
        input_data = pd.DataFrame([data.model_dump(by_alias=True)])
        input_data[["duration", "days_left"]] = si.transform(input_data[["duration", "days_left"]])
        input_data[["departure", "stops", "airline"]]=si_cat.transform(input_data[["departure", "stops", "airline"]])
        input_data = preprocessor.transform(input_data)
        prediction = model.predict(input_data)
        predicted_price = prediction[0] ** 2
        return {"predicted_price": float(predicted_price)}
    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )




@app.get("/health")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "API is running"} 

