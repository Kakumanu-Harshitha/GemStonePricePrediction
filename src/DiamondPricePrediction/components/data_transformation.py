import pandas as pd
import numpy as np
from src.DiamondPricePrediction.logger.logging import logging
from src.DiamondPricePrediction.exception import customexception

import os
import sys
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.DiamondPricePrediction.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    """Path to save the preprocessor object."""
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """Creates preprocessing pipelines and applies transformations."""

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        """Build preprocessing pipelines for numerical and categorical features."""
        try:
            logging.info('Data Transformation pipeline creation started')

            # Columns by type
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Ordered categories for ordinal encoding
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Creating numerical and categorical pipelines')

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[
                        cut_categories, color_categories, clarity_categories
                    ])),
                    ('scaler', StandardScaler())
                ]
            )

            # Combine both pipelines
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Exception occurred while creating transformation pipeline")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        """Load data, apply transformations, and save the preprocessor."""
        try:
            # Read datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded")
            logging.info(f"Train Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Head:\n{test_df.head().to_string()}")

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            # Split into input and target features
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Fit on train, transform both train & test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing on training and testing data")

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor object saved successfully")

            return train_arr, test_arr

        except Exception as e:
            logging.info("Exception occurred during data transformation process")
            raise customexception(e, sys)
