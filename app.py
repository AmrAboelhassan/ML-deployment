from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

model = None
# load the known model filename from the same directory as this script
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "random_forest_model.joblib")
le_item_path = os.path.join(base_dir, "le_item.joblib")
le_reason_path = os.path.join(base_dir, "le_reason.joblib")
if os.path.exists(model_path):
	try:
		model = joblib.load(model_path)
	except Exception:
		model = None
le_item = None
le_reason = None
if os.path.exists(le_item_path):
    try:
        le_item = joblib.load(le_item_path)
    except Exception:
        le_item = None
if os.path.exists(le_reason_path):
    try:
        le_reason = joblib.load(le_reason_path)
    except Exception:
        le_reason = None

@app.get("/")
def home():
	return {"status": "running on Fly.io"}

@app.post("/predict")
def predict(data: dict):
    try:
        if model is None or le_item is None or le_reason is None:
            raise HTTPException(status_code=503, detail="Model or LabelEncoders not loaded. Place 'model_clean.pkl', 'random_forest_model.joblib', 'le_item.joblib', and 'le_reason.joblib' in the app directory.")
        # Encode ITEM_CODE and RETURN_REASON_CODE
        item_code = data.get("ITEM_CODE")
        reason_code = data.get("RETURN_REASON_CODE")
        ordered_quantity = data.get("ORDERED_QUANTITY")
        return_value = data.get("RETURN_VALUE")
        item_encoded = le_item.transform([item_code])[0]
        reason_encoded = le_reason.transform([reason_code])[0]
        df = pd.DataFrame([[item_encoded, reason_encoded, ordered_quantity, return_value]],
                          columns=["ITEM_CODE", "RETURN_REASON_CODE", "ORDERED_QUANTITY", "RETURN_VALUE"])
        prediction = model.predict(df)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))