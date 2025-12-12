import os
import sys
import pandas as pd
from src.DiamondPricePrediction.exception import customexception
from src.DiamondPricePrediction.logger.logging import logging
from src.DiamondPricePrediction.utils.utils import load_object


class PredictPipeline:
    """
    Handles prediction: loads model + preprocessor and performs inference.
    """

    def __init__(self):
        print("PredictPipeline initialized")

    def predict(self, features):
        """
        Apply preprocessing and return model prediction.
        """
        try:
            # Paths to saved artifacts
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            # Load preprocessor and trained model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform input data and predict
            scaled_fea = preprocessor.transform(features)
            pred = model.predict(scaled_fea)

            return pred

        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    """
    Creates a structured DataFrame from user input for prediction.
    """

    def __init__(
        self,
        carat: float,
        depth: float,
        table: float,
        x: float,
        y: float,
        z: float,
        cut: str,
        color: str,
        clarity: str
    ):
        # Assign user inputs
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        """
        Convert input values into a pandas DataFrame.
        """
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Custom input converted to DataFrame')
            return df

        except Exception as e:
            logging.info('Exception occurred while creating DataFrame')
            raise customexception(e, sys)
