import sys
import numpy as np
import os
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self.expected_columns = None  # Store expected column structure
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df, reference_columns=None):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        
        # Get categorical columns that need dummies
        categorical_columns = ['Vehicle_Age', 'Vehicle_Damage']
        
        # Create dummies
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        # If reference columns are provided, ensure all expected columns exist
        if reference_columns is not None:
            for col in reference_columns:
                if col not in df.columns:
                    df[col] = 0  # Add missing column with zeros
            
            # Reorder columns to match reference
            df = df[reference_columns]
        
        return df

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        
        # Define column mapping
        column_mapping = {
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years",
            "Vehicle_Damage_Yes": "Vehicle_Damage_Yes"
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure expected columns exist
        expected_dummy_columns = ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]
        for col in expected_dummy_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing column with default value
            df[col] = df[col].astype('int')
            
        return df

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def _ensure_consistent_columns(self, df, reference_columns):
        """Ensure DataFrame has consistent columns with reference."""
        missing_cols = set(reference_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(reference_columns)
        
        # Add missing columns with default values
        for col in missing_cols:
            df[col] = 0
            
        # Remove extra columns
        df = df[reference_columns]
        
        return df

    def save_preprocessor(self, preprocessor):
        """Save preprocessor to the standard location"""
        try:
            # FIXED: Use the correct attribute name - data_transformation_dir
            main_path = os.path.join(self.data_transformation_config.data_transformation_dir, "preprocessor.pkl")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(main_path), exist_ok=True)
            save_object(main_path, preprocessor)
            logging.info(f"Preprocessor saved at: {main_path}")
            
            # Also save to transformed_object directory for compatibility
            object_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)
            os.makedirs(object_dir, exist_ok=True)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            logging.info(f"Preprocessor also saved at: {self.data_transformation_config.transformed_object_file_path}")
            
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Store original column structure for reference
            original_columns = train_df.columns.tolist()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations to training data first
            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self._rename_columns(input_feature_train_df)
            
            # Store the column structure from training data
            self.expected_columns = input_feature_train_df.columns.tolist()
            logging.info(f"Expected columns structure: {self.expected_columns}")

            # Apply same transformations to test data using training data as reference
            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df, self.expected_columns)
            input_feature_test_df = self._rename_columns(input_feature_test_df)
            input_feature_test_df = self._ensure_consistent_columns(input_feature_test_df, self.expected_columns)
            
            logging.info("Custom transformations applied to train and test data")
            logging.info(f"Train shape: {input_feature_train_df.shape}, Test shape: {input_feature_test_df.shape}")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            # Transform data
            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            # Ensure we're working with numpy arrays
            from scipy import sparse
            if sparse.issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if sparse.issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()
                
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority", random_state=42)
            
            # Use fit_resample only on training data
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
            # For test data, we don't apply SMOTEENN as it's for evaluation
            input_feature_test_final, target_feature_test_final = input_feature_test_arr, target_feature_test_df
            logging.info("SMOTEENN applied to train data only.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            # Save objects using the new method
            self.save_preprocessor(preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            # Also save the expected column structure
            column_structure_path = self.data_transformation_config.transformed_object_file_path.replace('.pkl', '_columns.pkl')
            save_object(column_structure_path, self.expected_columns)
            
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e