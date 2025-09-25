import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.exception import MyException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import DataValidationConfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.config_entity import ModelTrainerConfig
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.config_entity import ModelPusherConfig

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.artifact_entity import ModelPusherArtifact

# ADD THIS IMPORT
from src.pipline.setup_prediction import setup_latest_model

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

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
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Entering the start_data_validation method of TrainPipeline class")
            logging.info("Validating the dataset")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact, 
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, 
                                 data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Entering the data_transformation method of the TrainPipeline class")
            logging.info("Transforming the dataset")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact, 
                data_validation_artifact=data_validation_artifact, 
                data_transformation_config=self.data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Entering the start_model_trainer method of TrainPipeline")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training completed successfully")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                           model_trainer_artifact: ModelTrainerArtifact,
                           data_transformation_artifact: DataTransformationArtifact) -> ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(
             model_eval_config=self.model_evaluation_config,
             data_ingestion_artifact=data_ingestion_artifact,
             model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
    
    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            logging.info("Starting model pusher")
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Model pushing completed successfully")
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e                   
                                            
    def run_pipeline(self) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            logging.info("Starting training pipeline...")
            
            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data ingestion completed successfully!")
            
            # Data Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            logging.info("Data validation completed successfully!")
            
            # Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact, 
                data_validation_artifact=data_validation_artifact
            )
            logging.info("Data transformation completed successfully!")
            
            # Model Training
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            logging.info("Model training completed successfully!")
            
            # Model Evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact
            )
            
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted. Stopping pipeline.")
                return None
                
            # Model Pushing
            model_pusher_artifact = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )
            
            # ADD THIS: Set up the latest model for prediction
            logging.info("Setting up latest model for prediction...")
            if setup_latest_model():
                logging.info("Prediction model setup completed successfully!")
            else:
                logging.warning("Failed to set up prediction model")
            
            logging.info("Training pipeline completed successfully!")
            return model_pusher_artifact
            
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