import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class ModelEvaluationConfig:
    pass


class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            logging.info("Model evaluation started")

            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(file_path=model_path)

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model evaluation metrics - MAE: {mae}, RMSE: {rmse}, R2: {r2}")

            return {
                "mae": mae,
                "rmse": rmse,
                "r2_score": r2,
            }

        except Exception as e:
            raise CustomException(e, sys)
