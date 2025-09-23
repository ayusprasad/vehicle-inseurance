import os
from datetime import date

# For MongoDB connection
DATABASE_NAME = "vehicle_insurance"  # Changed from "Proj1" to match your actual database
COLLECTION_NAME = "insurance_data"   # Changed from "Proj1-Data" to match your actual collection
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = "vehicle_insurance_pipeline"  # Added a meaningful name
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "Response"  # Make sure this column exists in your data
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

"""
Data Ingestion related constants
"""
DATA_INGESTION_COLLECTION_NAME: str = "insurance_data"  # Fixed to match your collection
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

"""
Data Validation related constants
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"  # Added missing constant

"""
Data Transformation related constants
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
Model Trainer related constants
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_N_ESTIMATORS: int = 200  # Added type annotation
MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 7
MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6
MODEL_TRAINER_MAX_DEPTH: int = 10  # Fixed variable name
MODEL_TRAINER_CRITERION: str = 'entropy'  # Fixed variable name
MODEL_TRAINER_RANDOM_STATE: int = 101  # Fixed variable name

"""
Model Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "vehicle-insurance-models"  # Changed to a more specific name
MODEL_PUSHER_S3_KEY = "model-registry"

"""
Model Pusher related constants (added missing section)
"""
MODEL_PUSHER_DIR_NAME: str = "model_pusher"

"""
Application related constants
"""
APP_HOST = "0.0.0.0"
APP_PORT = 5000

"""
Added missing constants that are commonly needed
"""
SEED = 42  # For reproducibility
NUMERICAL_COLUMNS = ["Age", "Region_Code", "Annual_Premium", "Vintage"]  # Example, update based on your data
CATEGORICAL_COLUMNS = ["Gender", "Driving_License", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage"]  # Example, update based on your data