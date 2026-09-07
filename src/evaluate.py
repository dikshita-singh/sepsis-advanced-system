import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_sepsis_model():
    print("=== STARTING MODEL EVALUATION ===")
    
    # 1. Paths setup
    processed_dir = "data/processed"
    model_path = "models/sepsis_model.pkl"
    
    # 2. Load the evaluation test splits we saved earlier
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))
    
    # 3. Load the saved model brain
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print("Model asset and test data loaded successfully.")
    
    # 4. Generate Predictions
    y_pred = model.predict(X_test)
    
    # 5. Print Performance Evaluation Metrics
    print("\n--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (Healthy predicted Healthy): {cm[0][0]}")
    print(f"False Positives (Healthy predicted Sepsis):  {cm[0][1]}")
    print(f"False Negatives (Sepsis predicted Healthy):  {cm[1][0]}")
    print(f"True Positives (Sepsis predicted Sepsis):   {cm[1][1]}")
    
    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_sepsis_model()