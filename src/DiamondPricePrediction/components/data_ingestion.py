import pandas as pd
import numpy as np
from src.DiamondPricePrediction.logger.logging import logging
from src.DiamondPricePrediction.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """Paths for storing raw, train, and test data in artifacts folder."""
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    """Handles downloading, splitting, and saving raw data."""
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Reads dataset, performs train-test split, and saves outputs."""
        logging.info("Data ingestion started")

        try:
            # Load dataset from GitHub (raw file URL)
            data = pd.read_csv(
                "https://raw.githubusercontent.com/Kakumanu-Harshitha/GemStonePricePrediction/refs/heads/main/raw.csv"
            )
            logging.info("Dataset loaded successfully")

            # Create artifacts directory if not already present
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True
            )

            # Save raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw dataset saved")

            # Split dataset into train/test
            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train-test split completed")

            # Save train and test files
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occurred during data ingestion")
            raise customexception(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
