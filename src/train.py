import os
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib  #For saving the trained model file
import pickle
from src.config_loader import load_config

def train_sepsis_model():
    print("=== STARTING MODEL TRAINING ===")
    
    # 1. Load Configurations
    config = load_config()
    processed_data_path = os.path.join(config['paths']['processed_data_dir'], "sepsis_processed.csv")
    
    # 2. Load Processed Dataset
    df = pd.read_csv(processed_data_path)
    
    # Separate features (X) and target label (y)
    target = config['model']['target_column']
    
    # Drop columns that are IDs or administrative metrics, not clinical signs
    ignore_cols = [target, 'Patient_ID', 'Unnamed: 0']
    features = [col for col in df.columns if col not in ignore_cols]
    
    X = df[features]
    y = df[target]
    
    print(f"Training features list: {features}")
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state'],
        stratify=y  # Maintains the same sepsis ratio in both train and test splits!
    )
    
    # 4. Handle Imbalance with Class Weights
    # Calculate scale_pos_weight: total negative samples / total positive samples
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count
    print(f"Data balance check - Negative: {neg_count}, Positive: {pos_count}")
    print(f"Calculated class balance weight multiplier: {scale_weight:.2f}")
    
    # 5. Initialize and Train LightGBM Classifier
    print("Training LightGBM model...")
    model = LGBMClassifier(
        n_estimators=config['model']['n_estimators'],
        learning_rate=config['model']['learning_rate'],
        random_state=config['model']['random_state'],
        scale_pos_weight=scale_weight,  # Tells model to pay massive attention to rare sepsis cases
        n_jobs=-1  # Use all computer CPU processor cores for speed
    )
    
    model.fit(X_train, y_train)
    print("Model training completed successfully!")
    
    # 6. Save Trained Model to Disk
    model_dir = config['paths']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "sepsis_model.pkl")
    
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Saved trained model asset to: {model_save_path}")
    
    # Save the splits to local system temporarily so our evaluate script can use them next
    X_test.to_csv(os.path.join(config['paths']['processed_data_dir'], "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(config['paths']['processed_data_dir'], "y_test.csv"), index=False)

if __name__ == "__main__":
    train_sepsis_model()