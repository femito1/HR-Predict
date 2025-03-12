import sys
import pandas as pd
import xgboost as xgb
import pickle
import os


from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.utils import load_object


print(os.getcwd())


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:
            print("in predict_pipeline.py file")
            model_path = 'artifacts/xgb_model2.pkl'
            # preprocesor_path = 'artifacts/preprocessor.pkl'
            
            model = load_object(file_path=model_path)
            # preprocessor = load_object(file_path=preprocesor_path)

            ohe_feature_names = load_object("artifacts/ohe_feature_names.pkl")  # Expected OHE columns
            scaler_stats = load_object("artifacts/scaler_stats.pkl")  # Mean & scale

            scaler = StandardScaler()
            scaler.mean_ = scaler_stats["mean"]
            scaler.scale_ = scaler_stats["scale"]

            num_features = ["satisfaction_level", "last_evaluation", "number_project",
                            "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years"]
            cat_features = ["department", "salary"]


            ohe_df = pd.get_dummies(features[cat_features])
            
            # Ensure all expected one-hot encoded columns exist
            for col in ohe_feature_names:
                if col not in ohe_df.columns:
                    ohe_df[col] = 0  # Add missing columns with default value 0

            # Reorder columns to match training
            ohe_df = ohe_df[ohe_feature_names]

            # Step 2: Standard Scaling
            # features[num_features] = scaler.transform(features[num_features])

            # Step 3: Combine numerical & encoded categorical data
            features_processed = pd.concat([features[num_features], ohe_df], axis=1)

            # Step 4: Ensure correct column order (XGBoost needs exact same order)
            missing_cols = set(model.feature_names) - set(features_processed.columns)
            for col in missing_cols:
                features_processed[col] = 0  # Add missing columns

            features_processed = features_processed[model.feature_names]

            # Step 5: Predict using XGBoost
            predictions = model.predict(xgb.DMatrix(features_processed, feature_names = model.feature_names))

            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_from_csv(self, df):
            return df



class CustomData:
    def __init__(self, satisfaction_level, last_evaluation, number_project, average_montly_hours, 
                 time_spend_company, Work_accident, promotion_last_5years, department, salary):
        
        # THESE VALUES ARE COMING FROM WEB APPLICATION
        # NEED TO BE SAME AS WHAT IS USED IN THE .HTML FORM name="feat1"...
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
        #   This will just return all data as a DATAFRAME
        #   self means whatever we have from the webapp in that moment is being mapped to df values/features
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




