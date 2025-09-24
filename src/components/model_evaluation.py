import sys
import json
import os
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from dataclasses import dataclass
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.expected_columns = None
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[object]:
        """Get the best model from production (currently returns None)."""
        try:
            return None
        except Exception as e:
            raise MyException(e, sys)

    def _ensure_evaluation_report_path(self):
        """Ensure evaluation report path exists."""
        try:
            os.makedirs(os.path.dirname(self.model_eval_config.evaluation_report_path), exist_ok=True)
            return self.model_eval_config.evaluation_report_path
        except Exception as e:
            fallback_path = os.path.join("reports", "evaluation_report.json")
            os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
            return fallback_path

    def _save_evaluation_report(self, evaluate_model_response: EvaluateModelResponse, y_true, y_pred) -> str:
        """Save evaluation report to file."""
        try:
            report_path = self._ensure_evaluation_report_path()
            
            report = {
                "model_evaluation": {
                    "is_model_accepted": evaluate_model_response.is_model_accepted,
                    "trained_model_f1_score": evaluate_model_response.trained_model_f1_score,
                    "best_model_f1_score": evaluate_model_response.best_model_f1_score,
                    "difference": evaluate_model_response.difference
                },
                "classification_report": classification_report(y_true, y_pred, output_dict=True),
                "model_info": {
                    "trained_model_path": self.model_trainer_artifact.trained_model_file_path,
                }
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            logging.info(f"Evaluation report saved at: {report_path}")
            return report_path
            
        except Exception as e:
            logging.warning(f"Could not save evaluation report: {str(e)}")
            return "reports/evaluation_report_fallback.json"

    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df, reference_columns=None):
        """Create dummy variables for categorical features."""
        categorical_columns = ['Vehicle_Age', 'Vehicle_Damage']
        
        # Create dummies only for columns that exist
        existing_categorical = [col for col in categorical_columns if col in df.columns]
        if existing_categorical:
            df = pd.get_dummies(df, columns=existing_categorical, drop_first=True)
        
        # Ensure all expected columns exist
        if reference_columns is not None:
            for col in reference_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[reference_columns]
        
        return df

    def _rename_columns(self, df):
        """Rename specific columns."""
        column_mapping = {
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years",
            "Vehicle_Damage_Yes": "Vehicle_Damage_Yes"
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        return df

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def _load_expected_columns(self):
        """Load the expected column structure from training."""
        try:
            column_structure_path = self.data_transformation_artifact.transformed_object_file_path.replace('.pkl', '_columns.pkl')
            if os.path.exists(column_structure_path):
                return load_object(column_structure_path)
            return None
        except:
            return None

    def _prepare_features(self, X):
        """Prepare features using the same transformations as training."""
        try:
            # Load the preprocessor
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            
            # Load expected column structure
            expected_columns = self._load_expected_columns()
            
            # Apply the same transformations that were done during training
            X_transformed = self._map_gender_column(X)
            X_transformed = self._drop_id_column(X_transformed)
            X_transformed = self._create_dummy_columns(X_transformed, expected_columns)
            X_transformed = self._rename_columns(X_transformed)
            
            # Ensure column consistency
            if expected_columns is not None:
                missing_cols = set(expected_columns) - set(X_transformed.columns)
                for col in missing_cols:
                    X_transformed[col] = 0
                X_transformed = X_transformed[expected_columns]
            
            # Apply preprocessor
            X_processed = preprocessor.transform(X_transformed)
            
            # Convert to numpy array
            if hasattr(X_processed, 'toarray'):
                return X_processed.toarray()
            else:
                return np.array(X_processed)
                
        except Exception as e:
            logging.error(f"Error preparing features: {e}")
            raise MyException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        """Evaluate the trained model."""
        try:
            # Load test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            # Prepare features using the same pipeline as training
            x_prepared = self._prepare_features(x)

            # Load trained model
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded successfully.")
            
            # Make predictions
            y_pred = trained_model.predict(x_prepared)
            
            # Calculate F1 score
            trained_model_f1_score = f1_score(y, y_pred, average='weighted')
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            # Compare with best model (currently none)
            best_model_f1_score = None
            best_model = self.get_best_model()
            
            if best_model is not None:
                y_hat_best_model = best_model.predict(x_prepared)
                best_model_f1_score = f1_score(y, y_hat_best_model, average='weighted')
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            
            # Save evaluation report
            self._save_evaluation_report(result, y, y_pred)
            
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Initiate the model evaluation process."""
        try:
            logging.info("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            
            evaluate_model_response = self.evaluate_model()

            # Create model evaluation artifact
            model_config = {
                "model_path": self.model_trainer_artifact.trained_model_file_path,
                "model_type": "RandomForestClassifier",
                "evaluation_threshold": 0.6,
            }

            report_path = self._ensure_evaluation_report_path()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                changed_accuracy=evaluate_model_response.difference,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                evaluation_report_path=report_path,
                model_config=model_config
            )

            logging.info(f"Model evaluation completed successfully")
            return model_evaluation_artifact
            
        except Exception as e:
            raise MyException(str(e), sys)
        
if __name__=="__main__":
    print("main")