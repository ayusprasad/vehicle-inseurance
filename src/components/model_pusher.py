import sys
import os
import logging
import boto3
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError

from src.exception import MyException
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.utils.main_utils import load_object


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.s3_client = boto3.client('s3', region_name='us-east-1')
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_s3_bucket(self):
        """Create S3 bucket if it doesn't exist - FIXED for us-east-1"""
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.model_pusher_config.bucket_name)
            logging.info(f"Bucket {self.model_pusher_config.bucket_name} already exists")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    # FIX: For us-east-1, don't use LocationConstraint
                    self.s3_client.create_bucket(Bucket=self.model_pusher_config.bucket_name)
                    logging.info(f"Created bucket: {self.model_pusher_config.bucket_name}")
                    
                    # Wait for bucket to be ready
                    import time
                    time.sleep(2)
                    return True
                except ClientError as create_error:
                    logging.error(f"Failed to create bucket: {create_error}")
                    return False
            else:
                logging.error(f"Error checking bucket: {e}")
                return False

    def _check_s3_credentials(self):
        """Check if AWS credentials are available"""
        try:
            self.s3_client.list_buckets()
            return True
        except NoCredentialsError:
            logging.warning("AWS credentials not found. Skipping S3 upload.")
            return False
        except ClientError as e:
            logging.warning(f"AWS credentials error: {e}. Skipping S3 upload.")
            return False

    def _upload_file_to_s3(self, local_file_path, s3_key):
        """Upload a file to S3 bucket"""
        try:
            if not os.path.exists(local_file_path):
                logging.warning(f"File {local_file_path} does not exist. Skipping upload.")
                return False
                
            self.s3_client.upload_file(
                local_file_path,
                self.model_pusher_config.bucket_name,
                s3_key
            )
            logging.info(f"✅ Successfully uploaded {local_file_path} to S3")
            return True
        except Exception as e:
            logging.warning(f"Failed to upload {local_file_path} to S3: {str(e)}")
            return False

    def _upload_model_to_s3(self, model_path):
        """Upload model file to S3"""
        try:
            if not os.path.exists(model_path):
                logging.error(f"Model file {model_path} does not exist")
                return False

            # Create S3 key with versioning
            s3_model_key = f"models/{self.model_version}/{os.path.basename(model_path)}"
            return self._upload_file_to_s3(model_path, s3_model_key)
        except Exception as e:
            logging.error(f"Error uploading model to S3: {str(e)}")
            return False

    def _upload_artifacts_to_s3(self):
        """Upload all artifacts to S3"""
        try:
            artifacts_uploaded = False
            
            # Upload model file
            model_path = self.model_evaluation_artifact.trained_model_path
            if self._upload_model_to_s3(model_path):
                artifacts_uploaded = True

            # Upload evaluation report if it exists
            report_path = getattr(self.model_evaluation_artifact, 'evaluation_report_path', None)
            if report_path and os.path.exists(report_path):
                s3_report_key = f"reports/{self.model_version}/{os.path.basename(report_path)}"
                if self._upload_file_to_s3(report_path, s3_report_key):
                    artifacts_uploaded = True

            return artifacts_uploaded
        except Exception as e:
            logging.error(f"Error uploading artifacts to S3: {str(e)}")
            return False

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """Initiate model pushing to S3"""
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Starting model pusher process")
            
            # Check if model is accepted
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model was not accepted in evaluation stage. Skipping model pushing.")
                return ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path="",
                    model_version=self.model_version
                )

            # Check S3 credentials and upload if available
            s3_upload_success = False
            if self._check_s3_credentials():
                logging.info("AWS credentials found. Checking bucket...")
                
                # Create bucket if it doesn't exist
                if self._create_s3_bucket():
                    logging.info("Attempting to upload to S3.")
                    s3_upload_success = self._upload_artifacts_to_s3()
                else:
                    logging.error("Failed to create S3 bucket. Upload skipped.")
            else:
                logging.info("No AWS credentials found. Model will be saved locally only.")

            # Create S3 model path
            s3_model_path = f"s3://{self.model_pusher_config.bucket_name}/models/{self.model_version}/" if s3_upload_success else ""

            # Create model pusher artifact
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=s3_model_path,
                model_version=self.model_version
            )

            if s3_upload_success:
                logging.info(f"✅ Successfully uploaded model to S3: {s3_model_path}")
            else:
                logging.info("📁 Model artifacts saved locally (S3 upload skipped or failed)")

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            
            return model_pusher_artifact
            
        except Exception as e:
            logging.error(f"Error in model pusher: {str(e)}")
            raise MyException(e, sys) from e