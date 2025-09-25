import sys
import os  # ADD THIS MISSING IMPORT
import pandas as pd
import numpy as np
from src.entity.config_entity import VehiclePredictorConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object


class VehicleData:
    def __init__(self,
                Gender: str,
                Age: int,
                Driving_License: int,
                Region_Code: float,
                Previously_Insured: int,
                Vehicle_Age: str,
                Vehicle_Damage: str,
                Annual_Premium: float,
                Policy_Sales_Channel: float,
                Vintage: int):
        """
        Vehicle Data constructor - Accepts raw data and handles transformation
        """
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Damage = Vehicle_Damage
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage

        except Exception as e:
            raise MyException(e, sys) from e

    def _preprocess_data(self):
        """Preprocess the data to match training pipeline format"""
        try:
            # Create a dictionary with the raw data
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage]
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(input_data)
            
            # Apply the same transformations as in training
            df = self._apply_transformations(df)
            
            return df
            
        except Exception as e:
            raise MyException(e, sys) from e

    def _apply_transformations(self, df):
        """Apply the same transformations as in the training pipeline"""
        try:
            # Map Gender
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
            
            # Create dummy variables for categorical features
            df = pd.get_dummies(df, columns=['Vehicle_Age', 'Vehicle_Damage'], drop_first=True)
            
            # Rename columns to match training format
            column_mapping = {
                "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years",
                "Vehicle_Damage_Yes": "Vehicle_Damage_Yes"
            }
            
            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure all expected columns exist
            expected_columns = [
                "Gender", "Age", "Driving_License", "Region_Code", "Previously_Insured",
                "Annual_Premium", "Policy_Sales_Channel", "Vintage",
                "Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"
            ]
            
            # Add missing columns with default value 0
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training order
            df = df[expected_columns]
            
            return df
            
        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self) -> pd.DataFrame:
        """
        This function returns a preprocessed DataFrame from VehicleData class input
        """
        try:
            return self._preprocess_data()
        except Exception as e:
            raise MyException(e, sys) from e


class VehicleDataClassifier:
    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = None):
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            if prediction_pipeline_config is None:
                prediction_pipeline_config = VehiclePredictorConfig()
            self.prediction_pipeline_config = prediction_pipeline_config
            self.model = None
            self.preprocessor = None
            self._load_model_and_preprocessor()
        except Exception as e:
            raise MyException(e, sys) from e

    def _load_model_and_preprocessor(self):
        """Load the model and preprocessor with proper error handling"""
        try:
            logging.info("Loading model and preprocessor for prediction")
            
            # Load model
            model_path = self.prediction_pipeline_config.model_file_path
            if not os.path.exists(model_path):
                # Try alternative paths
                alt_paths = [
                    os.path.join("artifact", "latest_model", "model.pkl"),
                    os.path.join("saved_models", "model.pkl"),
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        logging.info(f"Using model from: {model_path}")
                        break
                else:
                    raise FileNotFoundError(f"Model file not found at {model_path} or alternative locations")
            
            self.model = load_object(model_path)
            logging.info(f"Model loaded from: {model_path}")
            
            # Load preprocessor
            preprocessor_path = self.prediction_pipeline_config.preprocessing_object_path
            if not os.path.exists(preprocessor_path):
                # Try alternative paths
                alt_paths = [
                    os.path.join("artifact", "latest_model", "preprocessor.pkl"),
                    os.path.join("saved_models", "preprocessor.pkl"),
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        preprocessor_path = alt_path
                        logging.info(f"Using preprocessor from: {preprocessor_path}")
                        break
                else:
                    # If preprocessor not found, we can still proceed without it for basic prediction
                    logging.warning(f"Preprocessor file not found at {preprocessor_path}. Will use manual preprocessing.")
                    self.preprocessor = None
                    return
            
            self.preprocessor = load_object(preprocessor_path)
            logging.info(f"Preprocessor loaded from: {preprocessor_path}")
            
        except Exception as e:
            logging.error(f"Error loading model or preprocessor: {e}")
            raise MyException(e, sys) from e

    def _preprocess_input(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Preprocess input data using the same preprocessor as training"""
        try:
            if self.preprocessor is not None:
                # Apply the same preprocessing as in training
                processed_data = self.preprocessor.transform(dataframe)
                
                # Convert to numpy array if it's a sparse matrix
                if hasattr(processed_data, 'toarray'):
                    return processed_data.toarray()
                else:
                    return np.array(processed_data)
            else:
                # If no preprocessor found, return the dataframe as numpy array
                logging.warning("Using manual preprocessing since preprocessor was not found")
                return dataframe.values
                
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, vehicle_data: VehicleData) -> dict:
        """
        Predict vehicle insurance response
        Returns: Prediction result with probabilities
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            
            # Get preprocessed DataFrame
            input_df = vehicle_data.get_vehicle_input_data_frame()
            logging.info(f"Input data shape: {input_df.shape}")
            logging.info(f"Input columns: {input_df.columns.tolist()}")
            
            # Preprocess the input
            processed_input = self._preprocess_input(input_df)
            logging.info(f"Processed input shape: {processed_input.shape}")
            
            # Make prediction
            prediction = self.model.predict(processed_input)
            prediction_proba = self.model.predict_proba(processed_input)
            
            # Get the result
            result = {
                'prediction': int(prediction[0]),
                'probability': float(prediction_proba[0][1]),  # Probability of class 1
                'class_0_probability': float(prediction_proba[0][0]),
                'class_1_probability': float(prediction_proba[0][1])
            }
            
            logging.info(f"Prediction result: {result}")
            logging.info("Exited predict method of VehicleDataClassifier class")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise MyException(e, sys) from e


# High-level function to predict vehicle insurance response
def predict_insurance_response(
    Gender: str,
    Age: int,
    Driving_License: int,
    Region_Code: float,
    Previously_Insured: int,
    Vehicle_Age: str,
    Vehicle_Damage: str,
    Annual_Premium: float,
    Policy_Sales_Channel: float,
    Vintage: int
) -> dict:
    """
    High-level function to predict vehicle insurance response
    """
    try:
        # Create vehicle data object
        vehicle_data = VehicleData(
            Gender=Gender,
            Age=Age,
            Driving_License=Driving_License,
            Region_Code=Region_Code,
            Previously_Insured=Previously_Insured,
            Vehicle_Age=Vehicle_Age,
            Vehicle_Damage=Vehicle_Damage,
            Annual_Premium=Annual_Premium,
            Policy_Sales_Channel=Policy_Sales_Channel,
            Vintage=Vintage
        )
        
        # Create classifier and predict
        classifier = VehicleDataClassifier()
        result = classifier.predict(vehicle_data)
        
        return result
        
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Example usage
    try:
        # Sample prediction
        result = predict_insurance_response(
            Gender="Male",
            Age=34,
            Driving_License=1,
            Region_Code=28.0,
            Previously_Insured=0,
            Vehicle_Age="< 1 Year",
            Vehicle_Damage="Yes",
            Annual_Premium=2630.0,
            Policy_Sales_Channel=26.0,
            Vintage=217
        )
        
        print("Prediction Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")