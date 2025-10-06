import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

class MongoDBClient:
    def __init__(self):
        self.connection_string = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("DATABASE_NAME", "vehicle_insurance")
        self.client = None
        self.database = None
        
    def connect(self):
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=15000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000
            )
            
            # Test the connection
            self.client.admin.command('ping')
            self.database = self.client[self.database_name]
            
            print("✅ MongoDB Atlas connection established!")
            return True
            
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            return False
    
    def get_database(self):
        if self.database is None:
            self.connect()
        return self.database
    
    def get_collection(self, collection_name):
        database = self.get_database()
        return database[collection_name]
    
    def insert_insurance_data(self, data):
        """Helper method to insert insurance data"""
        collection = self.get_collection("insurance_records")
        result = collection.insert_one(data)
        return result.inserted_id
    
    def get_insurance_data(self, query={}):
        """Helper method to get insurance data"""
        collection = self.get_collection("insurance_records")
        return list(collection.find(query))
    
    def close_connection(self):
        if self.client:
            self.client.close()

# Global instance
mongodb_client = MongoDBClient()