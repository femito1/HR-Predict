import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# MOST LIKELY THIS ISNT NEEDED FOR ONLY USING TRAINED MODEL USING PICKLE   
# def evaluate_models(xtrain,ytrain,xtest,ytest,models,param):
#     try:
#         report = {}
#         logging.info(f"In Utils")

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para=param[list(models.keys())[i]]

#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(xtrain,ytrain)

#             model.set_params(**gs.best_params_)
#             model.fit(xtrain,ytrain)

#             #model.fit(X_train, y_train)  # Train model

#             y_train_pred = model.predict(xtrain)

#             y_test_pred = model.predict(xtest)

#             train_model_score = r2_score(ytrain, y_train_pred)

#             test_model_score = r2_score(ytest, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)