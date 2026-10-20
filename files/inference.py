import os
import json
import joblib
import pickle
import pandas as pd
import re            
import numpy as np   



# model_fn yang akan load artifacts from the unzipped tar.gz

def model_fn(model_dir):
    """
    SageMaker runs this once when the endpoint starts up.
    It loads the model, preprocessor, and column metadata into memory.
    """
    print(f"Loading artifacts from {model_dir}")
    
    # Load Model
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    # Load Preprocessor
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
    
    # Load Metadata ( sebagai blueprint )
    with open(os.path.join(model_dir, "model_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
        
    expected_cols = metadata['numerical_columns'] + metadata['categorical_columns']
    
    # We return a dictionary containing everything we need for predictions
    return {
        "model": model,
        "preprocessor": preprocessor,
        "expected_cols": expected_cols
    }





 
# input_fn: Parse JSON DARI streamlit

def input_fn(request_body, request_content_type):
    """
    Parses the JSON payload sent by your EC2 Streamlit frontend.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        # We expect a payload structured as {"instances": [{...}]} 
        # as defined in our deploy_endpoint.ipynb smoke test
        if "instances" in data:
            return data["instances"]
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")








# predict_fn: melakukan transformasi dan prediksi 

def _parse_credit_age(s):
    """Helper function matching your original preprocessing logic"""
    if pd.isna(s) or s is None:
        return np.nan
    years = re.search(r'(\d+)\s+Year', str(s))
    months = re.search(r'(\d+)\s+Month', str(s))
    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0
    return y * 12 + m

def predict_fn(input_data, artifacts):
    """
    Applies your OOP preprocessing and returns the model prediction.
    """
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    expected_cols = artifacts["expected_cols"]
    
    # Convert incoming dictionary to DataFrame
    df_input = pd.DataFrame(input_data)
    
    # DATA CLEANING INJECKSI
    # Clean structural numerics just like in preprocessing.py
    str_num_cols = [
        'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly'
    ]
    for col in str_num_cols:
        if col in df_input.columns:
            # Strip non-numeric characters
            cleaned = df_input[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df_input[col] = pd.to_numeric(cleaned, errors='coerce')
            
    # Parse the tricky Credit_History_Age column
    if 'Credit_History_Age' in df_input.columns:
        df_input['Credit_History_Age'] = df_input['Credit_History_Age'].apply(_parse_credit_age)
    # -------------------------------
    
    # Ensure all expected columns exist, filling missing ones with None
    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = None
            
    # Strictly enforce the column order required by the preprocessor
    df_input = df_input[expected_cols]
    
    # Transform using your custom pipeline
    X_trans = preprocessor.transform(df_input)
    
    # Predict
    predictions = model.predict(X_trans)
    
    return predictions.tolist()






# output_fn: format response ke frontend

def output_fn(prediction, accept):
    """
    Packages the prediction back into a JSON response
    """
    if accept == "application/json":
        return json.dumps({"predictions": prediction}), accept
    raise Exception(f"Requested unsupported ContentType in Accept: {accept}")