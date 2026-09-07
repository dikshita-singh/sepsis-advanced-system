import os
import pandas as pd
from src.config_loader import load_config
from src.ingestion import load_raw_data
from src.preprocessing import handle_missing_values, drop_empty_columns
from src.feature_engineering import engineer_features

def run_data_pipeline():
    """
    Orchestrates the entire end-to-end data preparation sequence.
    """
    print("=== STARTING DATA PIPELINE ===")
    
    # 1. Load configurations
    config = load_config()
    raw_path = os.path.join(config['paths']['raw_data_dir'], "sepsis_raw.csv")
    processed_dir = config['paths']['processed_data_dir']
    
    # If raw path fails due to naming, fallback to look for Dataset.csv
    if not os.path.exists(raw_path):
        raw_path = os.path.join(config['paths']['raw_data_dir'], "Dataset.csv")
        
    # 2. Ingest Data
    df = load_raw_data(raw_path)
    
    # 3. Preprocess / Impute Missing Values
    df = handle_missing_values(df)
    
    # 4. Drop columns that are still totally empty
    threshold = config['preprocessing']['missing_threshold']
    df = drop_empty_columns(df, threshold=threshold)
    
    # 5. Engineer clinical features
    df = engineer_features(df)
    
    # 6. Save the final clean dataset
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "sepsis_processed.csv")
    df.to_csv(output_path, index=False)
    
    print(f"=== PIPELINE COMPLETE! Saved clean data to: {output_path} ===")

if __name__ == "__main__":
    run_data_pipeline()