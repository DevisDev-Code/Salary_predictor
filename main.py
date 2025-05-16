# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

# Load model and other assets
model = pickle.load(open("xgb_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
expected_columns = pd.read_csv("placement_synthetic_data_5000.csv").drop(columns=["PlacedOrNot"]).columns.tolist()

# Define input schema
class PlacementInput(BaseModel):
    data: dict

@app.get("/")
def root():
    return {"message": "Placement predictor API is running."}

@app.post("/predict")
def predict(input: PlacementInput):
    try:
        input_data = input.data

        # Convert input dict to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure all required columns are present
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

        # Apply label encoders
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # Predict
        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
