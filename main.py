from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and label encoders
model = pickle.load(open("xgb_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Define expected column order
expected_columns = [
    "gender", "cgpa", "stream", "work_experience", "internship_experience",
    "technical_proficiency", "communication_skills", "leadership_role",
    "num_projects", "competitions_won"
]

# Define columns that need label encoding
categorical_columns = [
    "gender", "stream", "leadership_role"
]

@app.post("/predict")
async def predict(request: Request):
    input_data = await request.json()

    try:
        df = pd.DataFrame([input_data])

        # Convert numeric fields explicitly
        df["cgpa"] = pd.to_numeric(df["cgpa"], errors="raise")
        df["technical_proficiency"] = pd.to_numeric(df["technical_proficiency"], errors="raise")
        df["communication_skills"] = pd.to_numeric(df["communication_skills"], errors="raise")
        df["num_projects"] = pd.to_numeric(df["num_projects"], errors="raise")
        df["competitions_won"] = pd.to_numeric(df["competitions_won"], errors="raise")

        # Apply label encoders to categorical columns
        for col in categorical_columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        # Ensure correct column order
        df = df[expected_columns]

        print("Final input to model:", df.to_dict(orient="records"))

        # Make prediction
        prediction = model.predict(df)[0]
        return {"predicted_package": float(prediction)}

    except Exception as e:
        return {"error": str(e)}
