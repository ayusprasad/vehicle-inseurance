import sys
import os
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method to initiate model pushing
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info("Starting model pusher")
            
            # Check if model is accepted
            if self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model accepted. Starting push to S3...")
                
                try:
                    import boto3
                    from botocore.exceptions import ProfileNotFound, NoCredentialsError
                    
                    # Try to use user1 profile, fallback to default
                    try:
                        session = boto3.Session(profile_name='user1')
                        logging.info("Using AWS profile: user1")
                    except ProfileNotFound:
                        logging.warning("Profile 'user1' not found. Using default AWS profile.")
                        session = boto3.Session()
                    
                    s3_client = session.client('s3')
                    
                    # Upload model file to S3
                    model_file_path = self.model_evaluation_artifact.trained_model_path
                    bucket_name = self.model_pusher_config.bucket_name
                    s3_model_key = self.model_pusher_config.s3_model_key_path
                    
                    if os.path.exists(model_file_path):
                        s3_client.upload_file(model_file_path, bucket_name, s3_model_key)
                        logging.info(f"Model uploaded to S3: s3://{bucket_name}/{s3_model_key}")
                        
                        # Also upload preprocessor if it exists
                        preprocessor_path = model_file_path.replace('model.pkl', 'preprocessor.pkl')
                        if os.path.exists(preprocessor_path):
                            s3_preprocessor_key = s3_model_key.replace('model.pkl', 'preprocessor.pkl')
                            s3_client.upload_file(preprocessor_path, bucket_name, s3_preprocessor_key)
                            logging.info(f"Preprocessor uploaded to S3: s3://{bucket_name}/{s3_preprocessor_key}")
                        
                        return ModelPusherArtifact(
                            bucket_name=bucket_name,
                            s3_model_path=f"s3://{bucket_name}/{s3_model_key}",
                            model_version="v1.0"
                        )
                    else:
                        logging.error(f"Model file not found: {model_file_path}")
                        return ModelPusherArtifact(
                            bucket_name=bucket_name,
                            s3_model_path="",
                            model_version="v1.0"
                        )
                        
                except NoCredentialsError:
                    logging.error("AWS credentials not found. Please configure AWS CLI.")
                    return ModelPusherArtifact(
                        bucket_name=bucket_name,
                        s3_model_path="",
                        model_version="v1.0"
                    )
                except Exception as e:
                    logging.error(f"Error pushing model to S3: {str(e)}")
                    return ModelPusherArtifact(
                        bucket_name=bucket_name,
                        s3_model_path="",
                        model_version="v1.0"
                    )
            else:
                logging.info("Model not accepted. Skipping model pushing.")
                return ModelPusherArtifact(
                    bucket_name="",
                    s3_model_path="",
                    model_version="v1.0"
                )
                
        except Exception as e:
            raise MyException(e, sys) from e