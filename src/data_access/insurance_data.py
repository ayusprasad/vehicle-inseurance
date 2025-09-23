import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME, COLLECTION_NAME
from src.exception import MyException
from src.logger import logging  # Added missing import

class InsuranceData:
    """
    A class to export MongoDB records as a pandas DataFrame for vehicle insurance data.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str = COLLECTION_NAME, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str, optional
            The name of the MongoDB collection to export. Defaults to COLLECTION_NAME constant.
        database_name : Optional[str]
            Name of the database (optional). Defaults to DATABASE_NAME.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.
        """
        try:
            # Access specified collection from the default or specified database
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]

            # Convert collection data to DataFrame and preprocess
            logging.info(f"Fetching data from MongoDB collection: {collection_name}")
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Data fetched with {len(df)} records and {len(df.columns)} columns")
            
            # Drop MongoDB's _id field if it exists
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            
            # Replace "na" strings with NaN
            df.replace({"na": np.nan, "": np.nan, "null": np.nan, "NULL": np.nan}, inplace=True)
            
            # Log basic info about the dataframe
            logging.info(f"DataFrame shape: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")
            
            return df

        except Exception as e:
            logging.error(f"Error exporting collection as DataFrame: {str(e)}")
            raise MyException(e, sys)
    
    def get_sample_data(self, n_samples: int = 5, collection_name: str = COLLECTION_NAME) -> pd.DataFrame:
        """
        Get a sample of data from the MongoDB collection.
        
        Parameters:
        ----------
        n_samples : int, optional
            Number of samples to return. Default is 5.
        collection_name : str, optional
            The name of the MongoDB collection. Defaults to COLLECTION_NAME.
            
        Returns:
        -------
        pd.DataFrame
            DataFrame containing sample data.
        """
        try:
            collection = self.mongo_client.database[collection_name]
            sample_data = list(collection.aggregate([{"$sample": {"size": n_samples}}]))
            df = pd.DataFrame(sample_data)
            
            # Drop MongoDB's _id field if it exists
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting sample data: {str(e)}")
            raise MyException(e, sys)
    
    def get_column_info(self, collection_name: str = COLLECTION_NAME) -> dict:
        """
        Get information about columns in the collection.
        
        Parameters:
        ----------
        collection_name : str, optional
            The name of the MongoDB collection. Defaults to COLLECTION_NAME.
            
        Returns:
        -------
        dict
            Dictionary with column information.
        """
        try:
            df = self.export_collection_as_dataframe(collection_name)
            
            column_info = {
                "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=[object]).columns.tolist(),
                "boolean_columns": df.select_dtypes(include=[bool]).columns.tolist(),
                "date_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
                "total_columns": len(df.columns),
                "total_records": len(df)
            }
            
            return column_info
            
        except Exception as e:
            logging.error(f"Error getting column info: {str(e)}")
            raise MyException(e, sys)

    def close_connection(self):
        """
        Close the MongoDB connection.
        """
        try:
            self.mongo_client.close_connection()
            logging.info("MongoDB connection closed successfully.")
        except Exception as e:
            logging.warning(f"Error closing connection: {str(e)}")