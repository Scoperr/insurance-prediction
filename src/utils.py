import sys
import os

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,X_test,y_train,y_test,models):
    report = {}
    for i in range(len(list(models))):
        model=list(models.values())[i]

        
        k_folds = KFold(n_splits = 5)
        score_lr=cross_val_score(model,X_train,y_train,cv=k_folds)
        mean_cv_score = np.mean(score_lr)
        std_cv_score = np.std(score_lr)

        model.fit(X_train,y_train)

        #Make Predictions
        y_pred=model.predict(X_test)

        # Evaluate the model
        mae, rmse, r2_square = evaluate_model(y_test, y_pred)
        report[list(models.keys())[i]] = r2_square

    return report
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square
        