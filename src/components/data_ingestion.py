# FOR READING CSV AND CALCULATING EMPLOYEE TURNOVER

# ALSO HERE WE WRITE THE FLOW OF DATA INGESTION -> TRANSFORMATION -> TRAINING
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging


if __name__ == '__main__':
    data_transformation = DataTransformation()
    train_data, preprocessor_path = data_transformation.initiate_data_transformation('HR.csv')

