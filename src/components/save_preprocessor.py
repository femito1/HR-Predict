import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from data_transformation import DataTransformation 
from src.exception import CustomException
from src.logger import logging


train_path = 'src/components/HR.csv' 

data_transformation = DataTransformation()

print(os.getcwd())  # Check the current working directory
print(os.path.exists("HR.csv"))  # Check if HR.csv exists


train_data, preprocessor_path = data_transformation.initiate_data_transformation(train_path)

print(f"Preprocessor saved at: {preprocessor_path}")
