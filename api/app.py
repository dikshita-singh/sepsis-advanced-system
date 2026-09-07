import os
import pickle
from turtle import st
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Sepsis Advanced Clinical API",
    description="Real-time production inference endpoint with NLP and Smart Actions.",
    version="2.0.0"
)

# 1. Pydantic Schema defined here (To avoid import errors and include 'notes')
class SepsisInputFeatures(BaseModel):
    HR: float
    O2Sat: float
    Temp: float
    SBP: float
    MAP: float
    DBP: float
    Resp: float
    notes: str = ""  # New NLP field

# 2. Original Safe Model Loading Logic
MODEL = None

@app.on_event("startup")
def load_model_asset():
    global MODEL
    model_path = os.path.join("models", "sepsis_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "models", "sepsis_model.pkl")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Critical Error: Trained model asset not found at {model_path}")
    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)
    print("=== MODEL LOADER: Sepsis model loaded into RAM memory successfully ===")

@app.get("/")
def home():
    return {"status": "ONLINE", "project": "Sepsis Advanced Detection API"}

# 3. NLP Logic
SEPSIS_KEYWORDS = ["shivering", "pale", "confused", "dizzy", "low urine", "chills", "clammy", "blood", "vomit", "nausea"]

def analyze_clinical_notes(notes: str):
    if not notes:
        return 0.0
    risk_penalty = 0.0
    notes_lower = notes.lower()
    for word in SEPSIS_KEYWORDS:
        if word in notes_lower:
            risk_penalty += 0.05
    return min(risk_penalty, 0.20)

# 4. Master Predict Endpoint (Combines 39-feature fix + NLP + Smart Actions)
@app.post("/predict")
def predict_sepsis(payload: SepsisInputFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not initialized or loaded yet.")
    
    try:
        # Convert to dictionary and extract text notes securely
        input_dict = payload.dict()
        notes_text = input_dict.pop('notes', "")  # Remove text before sending to ML model
        
        df = pd.DataFrame([input_dict])
        
        # Calculate engineered features
        eps = 1e-5
        df['shock_index'] = df['HR'] / (df['SBP'] + eps)
        df['pulse_pressure'] = df['SBP'] - df['DBP']
        
        # MAGIC FIX: Automatically fill missing 30 features with 0.0 to match training data
        if hasattr(MODEL, 'feature_name_'):
            model_features = MODEL.feature_name_
            for col in model_features:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Reorder columns to exactly match how the model was trained
            df = df[model_features]
        
        # Get base probability from ML model
        base_risk = float(MODEL.predict_proba(df)[0][1])
        
        # Add NLP text penalty
        nlp_penalty = analyze_clinical_notes(notes_text)
        final_risk = base_risk + nlp_penalty
        
        # Apply strict 0.35 Clinical Threshold
        is_sepsis = final_risk >= 0.35
        
        # Smart Actions Engine
        actions = []
        if is_sepsis:
            actions.append("🚨 URGENT: Initiate Sepsis 'Hour-1 Bundle' protocol.")
            if payload.SBP < 90 or df['shock_index'].iloc[0] > 0.8:
                actions.append("💧 Hemodynamic Alert: Start IV Fluids (30 ml/kg crystalloid).")
            if payload.O2Sat < 92:
                actions.append("🫁 Respiratory Alert: Initiate Oxygen Therapy.")
            if payload.Temp > 38.5:
                actions.append("💊 Infection Alert: Draw blood cultures and start antibiotics.")
        else:
            actions.append("✅ Patient stable. Monitor vital trends routinely.")

        # Return results to Frontend
        return {
            "final_risk_score": round(final_risk * 100, 2),
            "nlp_added_risk": round(nlp_penalty * 100, 2),
            "prediction": "Sepsis Confirmed" if is_sepsis else "Patient Stable",
            "actions": actions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Inference Failure: {str(e)}")