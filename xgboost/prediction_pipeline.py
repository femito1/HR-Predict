import pandas as pd
import xgboost as xgb
import os
from torch import tensor, float32

def predict_attrition_from_csv(csv_path, model_path='xgboost/xgboost_hr_attrition_model.json'):
    """
    Process a CSV file of employee data, make attrition predictions, and calculate turnover rate.
    
    Args:
        csv_path: Path to the CSV file containing employee data
        model_path: Path to the saved XGBoost model JSON file
        feature_names_path: Path to the saved feature names pickle file
        
    Returns:
        DataFrame with original data plus predictions and overall turnover rate
    """
    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_path)
    
    
    # Load and validate the CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {df.shape[0]} employees and {df.shape[1]} features.")
        print()
    except Exception as e:
        raise Exception(f"Error loading CSV: {str(e)}")
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    if 'sales' in df_processed.columns:
        df_processed = df_processed.rename(columns={'sales': 'department'})
    if 'left' in df_processed.columns:
        df_processed = df_processed.drop(columns=['left'])
    
    # Apply preprocessing steps (same as in the notebook)
    
    # 1. Handle missing values
    if df_processed.isnull().sum().sum() > 0:
        print(f"Warning: Found {df_processed.isnull().sum().sum()} missing values. Filling with appropriate values.")
        print()
        
        # Numeric columns: fill with median
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Categorical columns: fill with mode
        cat_cols = df_processed.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # 2. Convert categorical variables to numeric
    # One-hot encode categorical variables

    feature_names = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'department_IT', 'department_RandD',
       'department_accounting', 'department_hr', 'department_management',
       'department_marketing', 'department_product_mng', 'department_sales',
       'department_support', 'department_technical', 'salary_high',
       'salary_low', 'salary_medium']

    df_encoded = pd.get_dummies(df_processed)
    categorical_features =  ['department_IT', 'department_RandD', 'department_accounting', 'department_hr', 'department_management', 'department_marketing', 'department_product_mng', 'department_sales', 'department_support', 'department_technical', 'salary_high', 'salary_low', 'salary_medium']
    df_encoded[categorical_features] = df_encoded[categorical_features].astype(float)

    # 3. Ensure all expected features are present
    missing_features = [feat for feat in feature_names if feat not in df_encoded.columns]
    extra_features = [feat for feat in df_encoded.columns if feat not in feature_names]
    
    if missing_features:
        print(f"Warning: Missing expected features: {missing_features}")
        print()
        # Add missing features with zeros
        for feat in missing_features:
            df_encoded[feat] = 0
    
    if extra_features:
        print(f"Warning: Found extra features not in training data: {extra_features}")
        print()
        # These will be ignored in prediction
    
    # 4. Ensure features are in the correct order
    df_encoded = df_encoded[feature_names]
    data = tensor(df_encoded.to_numpy(), dtype=float32)

    # 5. Make predictions
    dmatrix = xgb.DMatrix(data, feature_names=feature_names)
    probabilities = xgb_model.predict(dmatrix)
    
    threshold = 0.595
    predictions = (probabilities > threshold).astype(int)
    
    # 6. Add predictions to the original dataframe
    df['left'] = predictions
    
    # 7. Calculate turnover rate
    turnover_rate = predictions.mean() * 100

    return {
        'dataframe': df,
        'turnover_rate': turnover_rate,
        'employees_leaving': int(predictions.sum()),
        'total_employees': len(predictions)
    }



ret = predict_attrition_from_csv('data/HR.csv')
print(ret['dataframe'])
print(ret['turnover_rate'])
print(ret['employees_leaving'])
print(ret['total_employees'])
