import pandas as pd
import xgboost as xgb
from io import StringIO


def predict_attrition_from_csv(csv, model_path='xgboost/xgboost_hr_attrition_model.json'):
    """
    Process a CSV file of employee data, make attrition predictions, and calculate turnover rate.
    
    Args:
        csv_path: Path to the CSV file containing employee data
        model_path: Path to the saved XGBoost model JSON file
        feature_names_path: Path to the saved feature names pickle file (optional)
        
    Returns:
        Dictionary with original data plus predictions and overall turnover rate
    """
    
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_path)
    
    file = StringIO(csv)
    df = pd.read_csv(file)
    
    if 'left' in df.columns:
        df = df.drop(columns=['left'])
    if 'sales' in df.columns:
        df = df.rename(columns={'sales': 'department'})

    df_processed = df.copy()
    

    numeric_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']
    cat_cols =  ['department_IT', 'department_RandD', 'department_accounting', 'department_hr', 'department_management', 'department_marketing', 'department_product_mng', 'department_sales', 'department_support', 'department_technical', 'salary_high', 'salary_low', 'salary_medium']
    
    # to handle missing values, I am filling with the median for numeric columns 
    # and the mode for categorical columns
    if df_processed.isnull().sum().sum() > 0:
        print(f"Warning: Found {df_processed.isnull().sum().sum()} missing values. Filling with appropriate values.")
        
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        for col in cat_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    df_encoded = pd.get_dummies(df_processed)
    categorical_cols =  ['department_IT', 'department_RandD', 'department_accounting', 'department_hr', 'department_management', 'department_marketing', 'department_product_mng', 'department_sales', 'department_support', 'department_technical', 'salary_high', 'salary_low', 'salary_medium']
    feature_names = numeric_cols + categorical_cols
    
    
    if feature_names:
        missing_features = [feat for feat in feature_names if feat not in df_encoded.columns]
        extra_features = [feat for feat in df_encoded.columns if feat not in feature_names]
        
        if missing_features:
            print(f"Warning: Missing expected features: {missing_features}")
            for feat in missing_features:
                df_encoded[feat] = 0
        
        if extra_features:
            print(f"Warning: Found extra features not in training data: {extra_features}")
        
        df_encoded = df_encoded[feature_names]
    

    dmatrix = xgb.DMatrix(df_encoded, feature_names=feature_names)

    
    probabilities = xgb_model.predict(dmatrix)
    
    threshold = 0.595
    predictions = (probabilities > threshold).astype(int)
    
    df['probability'] = probabilities

    turnover_rate = predictions.mean() * 100
    
    print(f"Processed {df.shape[0]} employees.")
    print(f"Predicted turnover rate: {turnover_rate:.2f}%")
    print(f"Number of employees predicted to leave: {predictions.sum()} out of {len(predictions)}")
    
    return {
        'dataframe': df,
        'turnover_rate': turnover_rate,
        'employees_leaving': int(predictions.sum()),
        'total_employees': len(predictions)
    }