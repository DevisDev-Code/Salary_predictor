from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
import pickle

app = FastAPI()

# Allow CORS (so frontend like Bolt can access this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
model = pickle.load(open("xgb_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Expected input columns
expected_columns = [
    "gender", "cgpa", "stream", "work_experience", "internship_experience",
    "technical_proficiency", "communication_skills", "leadership_role",
    "num_projects", "competitions_won"
]

@app.post("/predict")
async def predict(request: Request):
    input_data = await request.json()

    try:
        df = pd.DataFrame([input_data])

        # Ensure correct column order
        df = df[expected_columns]

        # Apply label encoding where necessary
        for col in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        prediction = model.predict(df)[0]
        return {"predicted_package": float(prediction)}

    except Exception as e:
        return {"error": str(e)}
