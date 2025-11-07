from fastapi import APIRouter
from model.base_model import LoanPrediction
import numpy as np
import pickle

router = APIRouter()

@router.post("/predict_personal_loan")
async def predict_personal_loan(data: LoanPrediction):
    # Load trained classification model
    with open('personal_loan_classification.pkl', 'rb') as f:
        model = pickle.load(f)

    # Prepare input data as 2D array
    new_data = np.array([[
        data.age,
        data.experience,
        data.income,
        data.family,
        data.zip_code,
        data.ccavg,
        data.education,
        data.mortgage,
        data.securities_account,
        data.cd_account,
        data.online,
        data.creditcard
    ]])

    # Predict class (0 or 1)
    prediction = model.predict(new_data)[0]

    # Predict probability (if model supports it)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(new_data)[0][1]
    else:
        probability = None

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 3) if probability is not None else None,
        "message": "Customer is likely to take a personal loan."
        if prediction == 1 else "Customer is unlikely to take a personal loan."
    }
