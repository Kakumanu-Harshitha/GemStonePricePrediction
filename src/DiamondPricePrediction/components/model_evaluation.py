import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.DiamondPricePrediction.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.DiamondPricePrediction.logger.logging import logging
from src.DiamondPricePrediction.exception import customexception


class ModelEvaluation:
    """Evaluate a trained model and log metrics (optionally register model in MLflow)."""

    def __init__(self):
        logging.info("Model evaluation started")

    def eval_metrics(self, actual, pred):
        """Compute RMSE, MAE and R2 metrics."""
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("Evaluation metrics computed")
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        """
        Load the trained model, predict on test data, log metrics to MLflow,
        and optionally register the model if MLflow backend supports registry.
        """
        try:
            # Split features and target from test array
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            # Load trained model artifact
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            logging.info("Model loaded for evaluation")

            # Determine MLflow tracking store type (file vs server)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"MLflow tracking uri scheme: {tracking_url_type_store}")

            # Start MLflow run and log metrics
            with mlflow.start_run():
                prediction = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, prediction)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # If MLflow backend supports model registry (not file store), register model
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

                logging.info("Metrics logged to MLflow")

        except Exception as e:
            logging.info("Exception occurred during model evaluation")
            raise customexception(e, sys)
