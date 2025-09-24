from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str
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
    accuracy_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str 
    metric_artifact: ClassificationMetricArtifact
    model_config: dict

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float  # Remove the duplicate - only keep this one
    s3_model_path: str 
    trained_model_path: str
    evaluation_report_path: str
    model_config: dict

@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
    model_version: str