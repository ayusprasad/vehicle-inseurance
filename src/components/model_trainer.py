import sys
import os  # ADD THIS IMPORT
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Trains a RandomForestClassifier and evaluates metrics.

        Returns:
            trained model, metric artifact
        """
        try:
            logging.info("Splitting train and test arrays into features and targets.")
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            # Validate criterion
            allowed_criteria = ['gini', 'entropy', 'log_loss']
            criterion_value = self.model_trainer_config.criterion
            if criterion_value not in allowed_criteria:
                logging.warning(f"Invalid criterion '{criterion_value}' provided. Defaulting to 'gini'.")
                criterion_value = 'gini'

            # Initialize model
            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                min_samples_split=self.model_trainer_config.min_samples_split,
                min_samples_leaf=self.model_trainer_config.min_samples_leaf,
                max_depth=self.model_trainer_config.max_depth,
                criterion=criterion_value,
                random_state=self.model_trainer_config.random_state
            )

            logging.info("Training RandomForestClassifier...")
            model.fit(x_train, y_train)
            logging.info("Model training completed.")

            # Predictions
            y_pred = model.predict(x_test)

            # Metrics (safe for binary/multiclass)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
                accuracy_score=accuracy
            )
            logging.info(f"Evaluation metrics: {metric_artifact}")
            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            logging.info("Loading transformed train and test data...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            # Load preprocessing object
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded successfully.")

            # Check expected accuracy
            train_accuracy = accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1]))
            if train_accuracy < self.model_trainer_config.expected_accuracy:
                msg = (f"Model accuracy {train_accuracy:.4f} is below the base score "
                       f"{self.model_trainer_config.expected_accuracy}")
                logging.error(msg)
                raise MyException(msg, sys)

            # Save the trained model
            save_object(self.model_trainer_config.trained_model_file_path, trained_model)
            logging.info(f"Trained model saved at {self.model_trainer_config.trained_model_file_path}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                model_config=self.model_trainer_config
            )
            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
        
if __name__=="__main__":
    print("done")