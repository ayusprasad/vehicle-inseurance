import sys
import os
import time
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

class LightningModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        ⚡ Lightning-fast model trainer with extreme optimization
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.training_start_time = None
        self.performance_metrics = {}

    def log_training_time(self, stage: str):
        """⚡ Ultra-fast timing logger"""
        if self.training_start_time:
            elapsed = time.time() - self.training_start_time
            logging.info(f"⚡ {stage} completed in {elapsed:.3f} seconds")
            self.performance_metrics[stage] = elapsed

    def get_lightning_model_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[object, ClassificationMetricArtifact]:
        """
        ⚡ Trains ultra-fast model with lightning speed
        """
        try:
            logging.info("🚀 Starting lightning-speed model training...")
            self.training_start_time = time.time()
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            # ⚡ Ultra-fast model selection for maximum speed
            model = ExtraTreesClassifier(
                n_estimators=100,  # Optimized for speed
                max_depth=15,      # Balanced depth
                min_samples_split=15,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,        # Use all cores
                verbose=0,
                bootstrap=True
            )

            logging.info("⚡ Training lightning-fast ExtraTrees...")
            model.fit(x_train, y_train)
            self.log_training_time("Model training")

            # ⚡ Ultra-fast predictions
            y_pred = model.predict(x_test)

            # ⚡ Lightning-fast metric calculation
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
            
            logging.info(f"⚡ Lightning model metrics: {metric_artifact}")
            self.log_training_time("Model evaluation")
            
            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """⚡ Lightning-speed model training pipeline"""
        logging.info("🚀 Starting lightning model trainer")
        
        try:
            # ⚡ Ultra-fast data loading
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            # ⚡ Train model at lightning speed
            trained_model, metric_artifact = self.get_lightning_model_and_report(train=train_arr, test=test_arr)

            # ⚡ Ultra-fast preprocessing object loading
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)

            # ⚡ Instant accuracy verification
            train_accuracy = accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1]))
            if train_accuracy < self.model_trainer_config.expected_accuracy:
                logging.warning(f"⚡ Model accuracy {train_accuracy:.4f} below expected {self.model_trainer_config.expected_accuracy}")

            # ⚡ Lightning-fast model serialization
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(trained_model, self.model_trainer_config.trained_model_file_path)
            logging.info(f"⚡ Model saved at lightning speed: {self.model_trainer_config.trained_model_file_path}")

            # ⚡ Performance summary
            total_time = time.time() - self.training_start_time
            performance_grade = "⚡ Lightning Fast" if total_time < 30 else "🚀 Fast"
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                model_config=self.model_trainer_config,
                performance_metrics={
                    "total_time": total_time,
                    "speed_grade": performance_grade,
                    "training_speed": f"{total_time:.2f}s"
                }
            )
            
            logging.info(f"⚡ Total training time: {total_time:.2f} seconds - {performance_grade}")
            
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e

# Replace with lightning version
ModelTrainer = LightningModelTrainer

if __name__=="__main__":
    print("⚡ Lightning model trainer ready for instant training!")