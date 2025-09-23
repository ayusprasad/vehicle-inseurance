from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    train_file_path: str  # Changed from trained_file_path for consistency
    test_file_path: str   # Changed from test_file_path (kept same)

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str
    # Added drift report path
    drift_report_file_path: Optional[str] = None

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str 
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    # Added accuracy score
    accuracy_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str 
    metric_artifact: ClassificationMetricArtifact
    # Added model configuration
    model_config: dict

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float
    s3_model_path: str 
    trained_model_path: str
    # Added evaluation report path
    evaluation_report_path: str
    model_config: dict

@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
    # Added model version
    model_version: str