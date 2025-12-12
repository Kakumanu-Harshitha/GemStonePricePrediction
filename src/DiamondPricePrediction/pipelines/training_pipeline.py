import os
import sys
from src.DiamondPricePrediction.logger.logging import logging
from src.DiamondPricePrediction.exception import customexception
import pandas as pd

from src.DiamondPricePrediction.components.data_ingestion import DataIngestion
from src.DiamondPricePrediction.components.data_transformation import DataTransformation
from src.DiamondPricePrediction.components.model_trainer import ModelTrainer
from src.DiamondPricePrediction.components.model_evaluation import ModelEvaluation

# Step 1: Load raw data and create train/test split
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# Step 2: Transform data (scaling + encoding)
data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initialize_data_transformation(
    train_data_path, test_data_path
)

# Step 3: Train model using transformed data
model_trainer_obj = ModelTrainer()
model_trainer_obj.initate_model_training(train_arr, test_arr)

# Step 4: Evaluate model performance
model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr, test_arr)
