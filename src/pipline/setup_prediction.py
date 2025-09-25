import os
import shutil
import logging
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.main_utils import load_object

def find_preprocessor(artifact_dir, latest_dir):
    """Intelligently find the preprocessor file"""
    possible_locations = [
        # Most common locations
        os.path.join(artifact_dir, latest_dir, "data_transformation", "preprocessor.pkl"),
        os.path.join(artifact_dir, latest_dir, "data_transformation", "transformed_object", "preprocessor.pkl"),
        os.path.join(artifact_dir, latest_dir, "preprocessor.pkl"),
        os.path.join(artifact_dir, latest_dir, "data_transformation", "preprocessing.pkl"),
        
        # Fallback: search entire data_transformation directory
        None  # This will trigger a search
    ]
    
    for location in possible_locations:
        if location and os.path.exists(location):
            return location
    
    # If not found in expected locations, search recursively
    data_transformation_dir = os.path.join(artifact_dir, latest_dir, "data_transformation")
    if os.path.exists(data_transformation_dir):
        for root, dirs, files in os.walk(data_transformation_dir):
            for file in files:
                if file.endswith('.pkl') and ('preprocessor' in file.lower() or 'preprocess' in file.lower()):
                    return os.path.join(root, file)
    
    return None

def find_model(artifact_dir, latest_dir):
    """Intelligently find the model file"""
    possible_locations = [
        os.path.join(artifact_dir, latest_dir, "model_trainer", "trained_model", "model.pkl"),
        os.path.join(artifact_dir, latest_dir, "model_trainer", "model.pkl"),
        os.path.join(artifact_dir, latest_dir, "model.pkl"),
        os.path.join(artifact_dir, latest_dir, "trained_model", "model.pkl"),
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            return location
    
    # Search recursively in model_trainer directory
    model_trainer_dir = os.path.join(artifact_dir, latest_dir, "model_trainer")
    if os.path.exists(model_trainer_dir):
        for root, dirs, files in os.walk(model_trainer_dir):
            for file in files:
                if file.endswith('.pkl') and ('model' in file.lower()):
                    return os.path.join(root, file)
    
    return None

def setup_latest_model():
    """Copy the latest trained model and preprocessor to a fixed location for prediction"""
    try:
        # Find the latest artifact directory
        artifact_dir = "artifact"
        if not os.path.exists(artifact_dir):
            logging.error("No artifact directory found")
            return False
        
        # Get all timestamped directories
        dirs = [d for d in os.listdir(artifact_dir) if os.path.isdir(os.path.join(artifact_dir, d))]
        if not dirs:
            logging.error("No model directories found")
            return False
        
        # Sort by creation time (newest first)
        dirs.sort(key=lambda x: os.path.getctime(os.path.join(artifact_dir, x)), reverse=True)
        latest_dir = dirs[0]
        
        logging.info(f"Latest model directory: {latest_dir}")
        
        # Create latest_model directory
        latest_model_dir = os.path.join(artifact_dir, "latest_model")
        os.makedirs(latest_model_dir, exist_ok=True)
        
        # Find and copy model
        model_source = find_model(artifact_dir, latest_dir)
        model_dest = os.path.join(latest_model_dir, "model.pkl")
        
        if model_source and os.path.exists(model_source):
            shutil.copy2(model_source, model_dest)
            logging.info(f"Model copied from: {model_source}")
        else:
            logging.error(f"Model file not found in any expected location")
            return False
        
        # Find and copy preprocessor
        preprocessor_source = find_preprocessor(artifact_dir, latest_dir)
        preprocessor_dest = os.path.join(latest_model_dir, "preprocessor.pkl")
        
        if preprocessor_source and os.path.exists(preprocessor_source):
            shutil.copy2(preprocessor_source, preprocessor_dest)
            logging.info(f"Preprocessor copied from: {preprocessor_source}")
        else:
            logging.error(f"Preprocessor file not found in any expected location")
            return False
        
        logging.info("Latest model setup completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error setting up latest model: {e}")
        return False

if __name__ == "__main__":
    setup_latest_model()