import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
import torch
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    # path to save the preprocessor model only

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation/preprocessing
        
        '''
        try:


            categorical_columns = ["department", "salary"]  
            numerical_columns = ["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years"]  

            
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path):

        try:
            train_df=pd.read_csv(train_path)
            train_df.rename(columns={"sales": "department"}, inplace=True)

            categorical_columns = ["department", "salary"]  
            numerical_columns = ["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years"]  

            target_column_name="left"
            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]


            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_cat = ohe.fit_transform(train_df[categorical_columns])

            # Save feature names from OHE
            ohe_feature_names = ohe.get_feature_names_out(categorical_columns).tolist()


            path_features = os.path.join('artifacts',"ohe_feature_names.pkl")
            save_object(
                file_path=path_features,
                obj=ohe_feature_names
                )
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()      


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                "train_arr",
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)