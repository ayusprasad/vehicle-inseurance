import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

# Import project utils / exceptions / config
from src.exception import MyException
from src.logger import logging as project_logger
from src.utils.main_utils import load_object
from src.entity.config_entity import VehiclePredictorConfig

# mirror the project's logger
logging = project_logger

# Default expected columns used if preprocessor gives no guidance
DEFAULT_EXPECTED_INPUT_COLUMNS = [
    "id",  # often present during training; safe to include as default
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
    # Binary columns for vehicle age and damage (common in your project)
    "Vehicle_Age_lt_1_Year",
    "Vehicle_Age_gt_2_Years",
    "Vehicle_Damage_Yes"
]


class VehicleData:
    """
    Container for single observation incoming from UI/form.
    Accepts multiple plausible encodings and normalizes them to columns.
    """

    def __init__(self,
                 Gender,
                 Age,
                 Driving_License,
                 Region_Code,
                 Previously_Insured,
                 Annual_Premium,
                 Policy_Sales_Channel,
                 Vintage,
                 # optional encodings: either a single categorical string or explicit binary flags
                 Vehicle_Age: Optional[str] = None,
                 Vehicle_Age_lt_1_Year: Optional[int] = None,
                 Vehicle_Age_gt_2_Years: Optional[int] = None,
                 Vehicle_Damage: Optional[int] = None):
        try:
            # keep raw inputs
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage

            # flexible vehicle age encoding
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
            self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years

            # vehicle damage: allow 0/1 or "Yes"/"No"
            self.Vehicle_Damage = Vehicle_Damage
        except Exception as e:
            raise MyException(e, sys) from e

    def _normalize_gender(self, g):
        # Accept numeric 0/1 or strings 'Male'/'Female' (case-insensitive)
        try:
            if pd.isna(g):
                return 1  # default male (match earlier behaviour)
            if isinstance(g, (int, np.integer, float, np.floating)):
                return int(g)
            s = str(g).strip().lower()
            if s in {"male", "m", "1", "true", "yes"}:
                return 1
            if s in {"female", "f", "0", "false", "no"}:
                return 0
            # fallback
            return 1
        except Exception:
            return 1

    def _normalize_vehicle_age_flags(self):
        # Resolve to two binary flags: lt_1 and gt_2
        lt1 = 0
        gt2 = 0
        # priority: explicit flags if provided
        if self.Vehicle_Age_lt_1_Year is not None:
            try:
                lt1 = int(bool(int(self.Vehicle_Age_lt_1_Year)))
            except Exception:
                lt1 = 0
        if self.Vehicle_Age_gt_2_Years is not None:
            try:
                gt2 = int(bool(int(self.Vehicle_Age_gt_2_Years)))
            except Exception:
                gt2 = 0

        # if explicit flags not provided, try to parse Vehicle_Age string
        if (self.Vehicle_Age_lt_1_Year is None and self.Vehicle_Age is not None) or \
           (self.Vehicle_Age_gt_2_Years is None and self.Vehicle_Age is not None):
            s = str(self.Vehicle_Age).strip().lower() if self.Vehicle_Age is not None else ""
            # common training categories: "< 1 Year", "1-2 Year", "> 2 Years" etc.
            if "<" in s and "1" in s:
                lt1 = 1
                gt2 = 0
            elif ">" in s and ("2" in s or "2+" in s):
                lt1 = 0
                gt2 = 1
            elif "1-2" in s or "1 to 2" in s or "1 2" in s:
                lt1 = 0
                gt2 = 0
            else:
                # if ambiguous, leave both 0
                pass

        return int(lt1), int(gt2)

    def _normalize_vehicle_damage(self, val):
        try:
            if val is None:
                return 0
            if isinstance(val, (int, np.integer, float, np.floating)):
                return int(bool(int(val)))
            s = str(val).strip().lower()
            if s in {"1", "true", "yes", "y"}:
                return 1
            if s in {"0", "false", "no", "n"}:
                return 0
            return 0
        except Exception:
            return 0

    def get_vehicle_input_data_frame(self) -> pd.DataFrame:
        """
        Return DataFrame with common columns that the preprocessor or model expects.
        This function aims to be conservative: create all reasonable columns and
        leave the preprocessor/classifier to select/align.
        """
        try:
            gender_value = self._normalize_gender(self.Gender)
            lt1, gt2 = self._normalize_vehicle_age_flags()
            damage_flag = self._normalize_vehicle_damage(self.Vehicle_Damage)

            input_data = {
                # id is included as 0 by default so saved preprocessor expecting id won't break
                "id": [0],
                "Gender": [gender_value],
                "Age": [int(self.Age) if self.Age is not None else 0],
                "Driving_License": [int(self.Driving_License) if self.Driving_License is not None else 0],
                "Region_Code": [self.Region_Code if self.Region_Code is not None else 0],
                "Previously_Insured": [int(self.Previously_Insured) if self.Previously_Insured is not None else 0],
                "Annual_Premium": [float(self.Annual_Premium) if self.Annual_Premium is not None else 0.0],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel if self.Policy_Sales_Channel is not None else 0],
                "Vintage": [int(self.Vintage) if self.Vintage is not None else 0],
                "Vehicle_Age_lt_1_Year": [lt1],
                "Vehicle_Age_gt_2_Years": [gt2],
                "Vehicle_Damage_Yes": [damage_flag]
            }

            df = pd.DataFrame(input_data)

            # Replace non-finite values
            df = df.replace([np.nan, np.inf, -np.inf], 0)

            # Ensure int dtype for binary columns
            for c in ["Gender", "Driving_License", "Previously_Insured",
                      "Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
                if c in df.columns:
                    df[c] = df[c].astype(int)

            return df

        except Exception as e:
            raise MyException(e, sys) from e


class VehicleDataClassifier:
    """
    Loads model + preprocessor and performs a safe prediction.
    """

    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = None):
        try:
            if prediction_pipeline_config is None:
                prediction_pipeline_config = VehiclePredictorConfig()
            self.config = prediction_pipeline_config
            self.model = None
            self.preprocessor = None
            self.expected_input_columns: List[str] = DEFAULT_EXPECTED_INPUT_COLUMNS.copy()
            self._load_model_and_preprocessor()
        except Exception as e:
            raise MyException(e, sys) from e

    def _find_existing_path(self, primary_path: str, alternatives: List[str]) -> str:
        if os.path.exists(primary_path):
            return primary_path
        for p in alternatives:
            if os.path.exists(p):
                return p
        return primary_path  # return primary; load will fail and be reported

    def _load_model_and_preprocessor(self):
        try:
            logging.info("Loading model and preprocessor for prediction")

            model_path = self._find_existing_path(
                self.config.model_file_path,
                [
                    os.path.join("artifact", "latest_model", "model.pkl"),
                    os.path.join("saved_models", "model.pkl"),
                ]
            )
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            self.model = load_object(model_path)
            logging.info(f"Model loaded from: {model_path} (type: {type(self.model)})")

            preprocessor_path = self._find_existing_path(
                self.config.preprocessing_object_path,
                [
                    os.path.join("artifact", "latest_model", "preprocessor.pkl"),
                    os.path.join("saved_models", "preprocessor.pkl"),
                ]
            )

            if not os.path.exists(preprocessor_path):
                logging.warning(f"Preprocessor file not found at {preprocessor_path}. Will continue with manual alignment.")
                self.preprocessor = None
                return

            self.preprocessor = load_object(preprocessor_path)
            logging.info(f"Preprocessor loaded from: {preprocessor_path} (type: {type(self.preprocessor)})")

            # Try to read expected input column names from preprocessor
            try:
                if hasattr(self.preprocessor, "feature_names_in_"):
                    cols = list(self.preprocessor.feature_names_in_)
                    self.expected_input_columns = cols
                    logging.info(f"Preprocessor.feature_names_in_ found: {cols}")
                else:
                    # get_feature_names_out may return transformed names; try to infer input names
                    if hasattr(self.preprocessor, "get_feature_names_out"):
                        cols = list(self.preprocessor.get_feature_names_out())
                        # If these contain transformed names rather than input features, fallback to default
                        if all(isinstance(x, str) for x in cols) and len(cols) > 0:
                            logging.info("Preprocessor.get_feature_names_out returned names; using them as expected columns.")
                            self.expected_input_columns = cols
            except Exception as e:
                logging.warning(f"Could not read expected columns from preprocessor: {e}")
        except Exception as e:
            logging.error(f"Error loading model or preprocessor: {e}")
            raise MyException(e, sys) from e

    def _align_dataframe_to_expected(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add any missing expected columns (with safe defaults) and reorder columns to match expected order.
        """
        try:
            # Ensure values are finite
            df = df.replace([np.nan, np.inf, -np.inf], 0)

            expected = self.expected_input_columns or DEFAULT_EXPECTED_INPUT_COLUMNS
            # if expected contains more elaborate transformed feature names, try to match common raw input columns
            # but the safe step is to ensure all expected columns exist. Missing columns get 0.
            for col in expected:
                if col not in df.columns:
                    df[col] = 0

            # Reorder to expected. If expected has duplicates or names not present, fallback to present columns.
            try:
                df = df[expected]
            except Exception:
                # fall back to selecting the intersection in the order of default expected list
                present = [c for c in expected if c in df.columns]
                if not present:
                    # no matched columns: just return dataframe as-is
                    return df
                df = df[present]

            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def _preprocess_input(self, dataframe: pd.DataFrame) -> np.ndarray:
        try:
            df_aligned = self._align_dataframe_to_expected(dataframe.copy())

            if self.preprocessor is not None:
                # Use the saved preprocessor
                processed = self.preprocessor.transform(df_aligned)
                if hasattr(processed, "toarray"):
                    processed = processed.toarray()
                return np.asarray(processed)
            else:
                # No preprocessor: use conservative manual preprocessing:
                #  - ensure numeric columns are numeric
                #  - ensure binary columns are int
                df_manual = df_aligned.copy()
                numeric_cols = ["Age", "Region_Code", "Annual_Premium", "Policy_Sales_Channel", "Vintage"]
                for c in numeric_cols:
                    if c in df_manual.columns:
                        df_manual[c] = pd.to_numeric(df_manual[c], errors="coerce").fillna(0)

                binary_cols = ["Gender", "Driving_License", "Previously_Insured",
                               "Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]
                for c in binary_cols:
                    if c in df_manual.columns:
                        df_manual[c] = df_manual[c].astype(int)

                # If id is present and model doesn't expect it, we'll leave it: _align_dataframe_to_expected should have matched
                return df_manual.values

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, vehicle_data: VehicleData) -> Dict[str, Any]:
        """
        Returns prediction dict:
            - prediction: 0/1
            - probability: probability of class 1 if available (float)
            - raw: optional more detailed model outputs
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier")
            input_df = vehicle_data.get_vehicle_input_data_frame()
            logging.info(f"Input DF shape: {input_df.shape}; columns: {input_df.columns.tolist()}")

            X = self._preprocess_input(input_df)
            logging.info(f"Processed input shape: {X.shape}")

            # model predict
            if self.model is None:
                raise MyException("Model is not loaded.", sys)

            pred = self.model.predict(X)
            pred_label = int(np.asarray(pred).reshape(-1)[0])

            prob_pos = None
            prob_neg = None

            # Try predict_proba
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)
                # In many binary classifiers, proba shape is (n_samples, 2) => [:,1] is class 1 prob
                try:
                    prob_pos = float(np.asarray(proba)[0, 1])
                    prob_neg = float(np.asarray(proba)[0, 0])
                except Exception:
                    # fallback: if shape unexpected, set None
                    prob_pos = None
                    prob_neg = None
            else:
                # Try decision_function -> map to probabilities using a logistic transform
                if hasattr(self.model, "decision_function"):
                    try:
                        score = self.model.decision_function(X)
                        # score may be array
                        s0 = float(np.asarray(score).reshape(-1)[0])
                        prob_pos = 1.0 / (1.0 + np.exp(-s0))
                        prob_neg = 1.0 - prob_pos
                    except Exception:
                        prob_pos = None
                        prob_neg = None

            result = {
                "prediction": pred_label,
                "probability": prob_pos if prob_pos is not None else (1.0 if pred_label == 1 else 0.0),
                "class_0_probability": prob_neg if prob_neg is not None else (1.0 if pred_label == 0 else 0.0),
                "class_1_probability": prob_pos if prob_pos is not None else (1.0 if pred_label == 1 else 0.0),
            }

            logging.info(f"Prediction result: {result}")
            logging.info("Exited predict method of VehicleDataClassifier")
            return result

        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise MyException(e, sys) from e
