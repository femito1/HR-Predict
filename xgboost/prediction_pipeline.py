def predict_attrition(employee_data):
    """
    Make attrition predictions for new employee data.
    
    Args:
        employee_data: A dictionary or pandas DataFrame with employee features
                      (must have the same features as your training data)
        
    Returns:
        Dictionary with prediction probability and binary prediction
    """
    if isinstance(employee_data, dict):
       employee_data = pd.DataFrame([employee_data])
    

    employee_data_encoded = pd.get_dummies(employee_data)


    for col in feature_names:
        if col not in employee_data_encoded.columns:
            employee_data_encoded[col] = 0

    employee_data_encoded = employee_data_encoded[feature_names]


    dmatrix = xgb.DMatrix(employee_data_encoded, feature_names=feature_names)

    probability = xgb_model.predict(dmatrix)[0]
    prediction = 1 if probability > 0.37 else 0
    
    return {
        "probability": float(probability),
        "prediction": int(prediction),
        "will_leave": bool(prediction == 1)
    }
