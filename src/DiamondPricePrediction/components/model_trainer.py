import pandas as pd
import numpy as np
from src.DiamondPricePrediction.logger.logging import logging
from src.DiamondPricePrediction.exception import customexception
import os
import sys
from dataclasses import dataclass

from src.DiamondPricePrediction.utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass
class ModelTrainerConfig:
    """Path to save the trained model."""
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """Trains multiple regression models and selects the best one."""

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting input and target features for train and test sets')

            # Split into X and y
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Models to evaluate
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet()
            }

            # Evaluate all models
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')

            # Select best model based on R2 score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name}  |  R2 Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model: {best_model_name}  |  R2 Score: {best_model_score}')

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred during model training')
            raise customexception(e, sys)
