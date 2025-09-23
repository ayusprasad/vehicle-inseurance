import os
import sys
import pymongo
import certifi
from typing import Optional

from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

# Load the certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBClient:
    """
    MongoDBClient is responsible for establishing a connection to the MongoDB database.

    Attributes:
    ----------
    client : MongoClient
        A shared MongoClient instance for the class.
    database : Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    -------
    __init__(database_name: str) -> None
        Initializes the MongoDB connection using the given database name.
    """

    client = None  # Shared MongoClient instance across all MongoDBClient instances

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        """
        Initializes a connection to the MongoDB database. If no existing connection is found, it establishes a new one.

        Parameters:
        ----------
        database_name : str, optional
            Name of the MongoDB database to connect to. Default is set by DATABASE_NAME constant.

        Raises:
        ------
        MyException
            If there is an issue connecting to MongoDB or if the environment variable for the MongoDB URL is not set.
        """
        try:
            # Check if a MongoDB client connection has already been established; if not, create a new one
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)  # Retrieve MongoDB URL from environment variables
                
                if mongo_db_url is None:
                    # If not found in environment, try to use the direct connection string
                    # Update this with your actual MongoDB connection string
                    mongo_db_url = "mongodb+srv://ayush210prasad_db_user:LgvjaRaelXiqE4a1@cluster0.dzwccg1.mongodb.net/"
                    logging.warning(f"MongoDB URL not found in environment variable '{MONGODB_URL_KEY}'. Using direct connection string.")
                
                # Establish a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                
                # Test the connection
                MongoDBClient.client.admin.command('ismaster')
                logging.info("MongoDB connection established successfully.")
            
            # Use the shared MongoClient for this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name]  # Connect to the specified database
            self.database_name = database_name
            
        except pymongo.errors.ServerSelectionTimeoutError as e:
            logging.error(f"Cannot connect to MongoDB server: {e}")
            raise MyException(f"Cannot connect to MongoDB server: {e}", sys)
        except pymongo.errors.ConfigurationError as e:
            logging.error(f"MongoDB configuration error: {e}")
            raise MyException(f"MongoDB configuration error: {e}", sys)
        except Exception as e:
            logging.error(f"Unexpected error connecting to MongoDB: {e}")
            raise MyException(f"Unexpected error connecting to MongoDB: {e}", sys)
    
    def get_collection(self, collection_name: str):
        """
        Get a specific collection from the connected database.
        
        Parameters:
        ----------
        collection_name : str
            Name of the collection to retrieve.
            
        Returns:
        -------
        Collection
            The requested MongoDB collection.
        """
        try:
            return self.database[collection_name]
        except Exception as e:
            logging.error(f"Error accessing collection {collection_name}: {e}")
            raise MyException(f"Error accessing collection {collection_name}: {e}", sys)
    
    def close_connection(self):
        """
        Close the MongoDB connection.
        """
        if MongoDBClient.client:
            MongoDBClient.client.close()
            MongoDBClient.client = None
            logging.info("MongoDB connection closed.")