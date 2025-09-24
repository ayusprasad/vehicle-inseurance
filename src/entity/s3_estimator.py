from typing import Optional
from pandas import DataFrame
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
import sys


class Proj1Estimator:
    """
    Save/retrieve ML model from S3 and run predictions.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        :param bucket_name: Name of your S3 bucket
        :param model_path: Path to model inside the bucket
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: Optional[MyModel] = None

    def is_model_present(self) -> bool:
        """Check if the model exists in the bucket."""
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=self.model_path
            )
        except MyException as e:
            print(e)
            return False

    def load_model(self) -> MyModel:
        """Load the model from S3."""
        model = self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
        if not isinstance(model, MyModel):
            raise MyException(f"Expected MyModel, got {type(model)}", sys)
        return model

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """Upload a local model file to S3."""
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: DataFrame):
        """Run predictions using the stored model."""
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise MyException(e, sys)
