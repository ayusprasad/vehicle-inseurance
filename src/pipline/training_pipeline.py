import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.exception import MyException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
# from src.components.data_validation import DataValidation
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
# from src.components.model_evaluation import ModelEvaluation
# from src.components.model_pusher import ModelPusher

from src.entity.config_entity import DataIngestionConfig
# from src.entity.config_entity import DataValidationConfig
# from src.entity.config_entity import DataTransformationConfig
# from src.entity.config_entity import ModelTrainerConfig
# from src.entity.config_entity import ModelEvaluationConfig
# from src.entity.config_entity import ModelPusherConfig

from src.entity.artifact_entity import DataIngestionArtifact
# from src.entity.artifact_entity import DataValidationArtifact
# from src.entity.artifact_entity import DataTransformationArtifact
# from src.entity.artifact_entity import ModelTrainerArtifact
# from src.entity.artifact_entity import ModelEvaluationArtifact
# from src.entity.artifact_entity import ModelPusherArtifact

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        # self.data_validation_config = DataValidationConfig()
        # self.data_transformation_config = DataTransformationConfig()
        # self.model_trainer_config = ModelTrainerConfig()
        # self.model_evaluation_config = ModelEvaluationConfig()
        # self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            logging.info("Starting training pipeline...")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data ingestion completed successfully!")
            
            # Uncomment these as you progress
            # data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            # data_transformation_artifact = self.start_data_transformation(
            #     data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact=data_validation_artifact)
            # model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            # model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
            #                                                         model_trainer_artifact=model_trainer_artifact)
            # if not model_evaluation_artifact.is_model_accepted:
            #     logging.info(f"Model not accepted.")
            #     return None
            # model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            
            logging.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise MyException(e, sys)

# Add this to actually run the pipeline when the script is executed directly
if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Failed to run pipeline: {str(e)}")