import sys
import pandas as pd
import os
import numpy as np
from src.pipeline.pytorch_gradient_boosting import PyTorchGradientBoosting
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from src.exception import CustomException
from src.utils import load_object


print(os.getcwd())


class PredictPipeline:
    def __init__(self):
        self.model = None

    def predict(self, features):

        try:
            print("in predict_pipeline.py file")
            model_path = 'artifacts/xgboost.pth'
            model = PyTorchGradientBoosting.load_model(model_path)
            self.model = model
            cat_features = ["department", "salary"]
            ohe_df = pd.get_dummies(features[cat_features])
            num_features = ["satisfaction_level", "last_evaluation", "number_project",
                            "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years"]
            features_processed = pd.concat([features[num_features], ohe_df], axis=1)
            feature_names = num_features + ['department_IT', 'department_RandD', 'department_accounting', 'department_hr', 
                'department_management', 'department_marketing', 'department_product_mng', 
                'department_sales', 'department_support', 'department_technical', 
                'salary_high', 'salary_low', 'salary_medium']
            
            for feature in feature_names:
                if feature not in features_processed.columns:
                    features_processed[feature] = 0
            features_processed = features_processed[feature_names]
            predictions = model.predict_proba_calibrated(np.array(features_processed, dtype=np.float32))

            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_from_csv(self, dataframes):
        """
        Process a CSV file of employee data, make attrition predictions, and calculate turnover rate.
        
        Args:
            csv_path: Path to the CSV file containing employee data
            model_path: Path to the saved XGBoost model JSON file
            feature_names_path: Path to the saved feature names pickle file (optional)
            
        Returns:
            Dictionary with original data plus predictions and overall turnover rate
        """
        
        model_path = 'artifacts/xgboost.pth'
        xgb_model = PyTorchGradientBoosting.load_model(model_path)

        storage_t_rate =  []
        storage_df = []

        for df in dataframes:
        
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
            

            dmatrix = np.array(df_encoded, dtype=np.float32)

            
            probabilities = xgb_model.predict_proba_calibrated(dmatrix)
            
            threshold = 0.55
            predictions = np.array(probabilities > threshold, dtype=np.int32)
            
            df['probability'] = probabilities

            turnover_rate = predictions.mean() * 100
            
            print(f"Processed {df.shape[0]} employees.")
            print(f"Predicted turnover rate: {turnover_rate:.2f}%")
            print(f"Number of employees predicted to leave: {predictions.sum()} out of {len(predictions)}")
            
            storage_df.append(df)
            storage_t_rate.append(turnover_rate)

        return storage_df, storage_t_rate
    
    def suggest_improvements(self, employee: pd.DataFrame):
        means = {'satisfaction_level': 0.4400980117614114,
                'last_evaluation': 0.7181125735088211,
                'average_montly_hours': 207.41921030523662}
        suggestions = []
        y_pred = self.predict(employee)

        num_features = ["satisfaction_level", "last_evaluation", "number_project",
                            "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years"]
        
        feature_names = num_features + ['department_IT', 'department_RandD', 'department_accounting', 'department_hr', 
                'department_management', 'department_marketing', 'department_product_mng', 
                'department_sales', 'department_support', 'department_technical', 
                'salary_high', 'salary_low', 'salary_medium']
            
        for feature in feature_names:
            if feature not in employee.columns:
                employee[feature] = 0

        employee = employee[feature_names]

        employee_features = dict(zip(employee[['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'salary_low', 'salary_medium', 'salary_high']].columns.tolist(), *employee[['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'salary_low', 'salary_medium', 'salary_high']].values.tolist()))
        employee = employee.to_numpy()
        if y_pred > 0.55:
            if employee_features['satisfaction_level'] < means['satisfaction_level']:
                suggestions.append('It seems that this employee is not satisfied with their job, as their satisfaction level is lower than expected. We recommend that HR and managers talk to them to improve their situation.')
            if employee_features['last_evaluation'] > means['last_evaluation']:
                suggestions.append('This employee is performing very well. We have found that when employees have a very high last evaluation score, they are more likely to leave the company. We recommend that they get rewarded for their performance.')
            if employee_features['average_montly_hours'] > means['average_montly_hours']:
                suggestions.append('This employee is working a lot of hours. Work-life balance is important for employee retention, a few less hours a month could help them to be happier at work. Reconsidering your current resource planning and expanding your current workforce could be a good idea.')
            if employee_features['salary_low']:
                suggestions.append('This employee is earning a low salary. We recommend that you consider increasing their salary or offer them bonuses to retain them.')
        return suggestions
    
    def suggest_improvements_batch(self, employees):
        suggestions = []
        means = {'satisfaction_level': 0.4400980117614114,
                'last_evaluation': 0.7181125735088211,
                'average_montly_hours': 207.41921030523662}
        feats = ['satisfaction_level', 'last_evaluation', 'average_montly_hours']
        mews = []

        model_path = 'artifacts/xgboost.pth'
        xgb_model = PyTorchGradientBoosting.load_model(model_path)

        for df in employees:
            
            sugs = []
            mews = df[feats].mean().to_dict()

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
            

            dmatrix = np.array(df_encoded, dtype=np.float32)

            
            probabilities = xgb_model.predict_proba_calibrated(dmatrix)
        
            probabilities = np.array(probabilities, dtype=np.float32)
            y_pred = probabilities.mean()

            if y_pred > 0.55:
                if mews['satisfaction_level'] < means['satisfaction_level']:
                    sugs.append('It seems that a lot of employees are not satisfied with their job. We recommend that HR and managers talk to them to improve their situation.')
                if mews['last_evaluation'] > means['last_evaluation']:
                    sugs.append('A lot of employees have a very high last evaluation score. We recommend that you reward them for their performance.')
                if mews['average_montly_hours'] > means['average_montly_hours']:
                    sugs.append('A lot of employees are working a lot of hours. Work-life balance is important for employee retention, a few less hours a month could help them to be happier at work. Reconsidering your current resource planning and expanding your current workforce could be a good idea.')
            suggestions.append(sugs)

        return suggestions


class CustomData:
    def __init__(self, satisfaction_level, last_evaluation, number_project, average_montly_hours, 
                 time_spend_company, Work_accident, promotion_last_5years, department, salary):
        
        self.satisfaction_level = float(satisfaction_level)
        self.last_evaluation = float(last_evaluation)
        self.number_project = int(number_project)
        self.average_montly_hours = int(average_montly_hours)
        self.time_spend_company = int(time_spend_company)
        self.Work_accident = int(Work_accident)  
        self.promotion_last_5years = int(promotion_last_5years)  
        self.department = str(department)  
        self.salary = str(salary)

    def get_data_as_dataframe(self):
        try:
            custom_data_as_input_dict = {
                "satisfaction_level": [self.satisfaction_level],
                "last_evaluation": [self.last_evaluation],
                "number_project": [self.number_project],
                "average_montly_hours": [self.average_montly_hours],
                "time_spend_company": [self.time_spend_company],
                "Work_accident": [self.Work_accident],
                "promotion_last_5years": [self.promotion_last_5years],
                "department": [self.department],
                "salary": [self.salary]
            } 
        #   FROM WEBAPP ALL INPUTS ARE MAPPED TO THIS PARTICULAR VALUE/FORMAT
            return pd.DataFrame(custom_data_as_input_dict)

        except Exception as e:
            raise CustomException(e,sys)




