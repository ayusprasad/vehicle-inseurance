import boto3
from src.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union, List
import os, sys
from src.logger import logging
from mypy_boto3_s3.service_resource import Bucket, ObjectSummary
from src.exception import MyException
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv
import pickle


class SimpleStorageService:
    """
    A class for interacting with AWS S3 storage, providing methods for file management,
    data uploads, and data retrieval in S3 buckets.
    """

    def __init__(self):
        """
        Initializes the SimpleStorageService instance with S3 resource and client
        from the S3Client class.
        """
        try:
            s3_client = S3Client()
            self.s3_resource = s3_client.s3_resource
            self.s3_client = s3_client.s3_client
        except Exception as e:
            raise MyException(e, sys) from e

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        """Check if a specified S3 key path exists in the bucket."""
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_object(object_: object, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str, bytes]:
        """
        Reads the specified S3 object with optional decoding and formatting.
        Args:
            object_ (object): The S3 object.
            decode (bool): Whether to decode the object content as a string.
            make_readable (bool): Whether to convert content to StringIO for DataFrame usage.
        """
        try:
            raw_data = object_.get()["Body"].read()
            if decode:
                raw_data = raw_data.decode()
            return StringIO(raw_data) if make_readable else raw_data
        except Exception as e:
            raise MyException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        """Retrieve the S3 bucket object."""
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_file_object(self, filename: str, bucket_name: str) -> List[ObjectSummary]:
        """Retrieve all matching file objects from the bucket."""
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
            return file_objects
        except Exception as e:
            raise MyException(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """Load a serialized model from S3."""
        try:
            model_file = f"{model_dir}/{model_name}" if model_dir else model_name
            file_objects = self.get_file_object(model_file, bucket_name)
            if not file_objects:
                raise FileNotFoundError(f"Model {model_file} not found in bucket {bucket_name}")
            file_object = file_objects[0]
            model_obj = self.read_object(file_object, decode=False)
            return pickle.loads(model_obj)
        except Exception as e:
            raise MyException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """Create a folder in S3 if it doesn't exist."""
        try:
            folder_key = folder_name.rstrip("/") + "/"
            obj = self.s3_resource.Object(bucket_name, folder_key)
            try:
                obj.load()
                logging.info(f"Folder {folder_key} already exists in {bucket_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    self.s3_client.put_object(Bucket=bucket_name, Key=folder_key)
                    logging.info(f"Folder {folder_key} created in {bucket_name}")
                else:
                    raise
        except Exception as e:
            raise MyException(e, sys) from e

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        """Upload a local file to S3 with optional removal."""
        try:
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(f"Uploaded {from_filename} to {bucket_name}/{to_filename}")

            if remove:
                os.remove(from_filename)
                logging.info(f"Removed local file {from_filename} after upload")
        except Exception as e:
            raise MyException(e, sys) from e

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:
        """Upload a DataFrame as a CSV to S3."""
        try:
            data_frame.to_csv(local_filename, index=False, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:
        """Convert an S3 object to a DataFrame."""
        try:
            content = self.read_object(object_, make_readable=True)
            return read_csv(content, na_values="na")
        except Exception as e:
            raise MyException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """Read a CSV from S3 into a DataFrame."""
        try:
            file_objects = self.get_file_object(filename, bucket_name)
            if not file_objects:
                raise FileNotFoundError(f"{filename} not found in {bucket_name}")
            return self.get_df_from_object(file_objects[0])
        except Exception as e:
            raise MyException(e, sys) from e
