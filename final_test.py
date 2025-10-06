from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def final_test():
    print("🎯 Final MongoDB Atlas Test")
    print("=" * 40)
    
    uri = os.getenv("MONGODB_URI")
    
    try:
        client = MongoClient(uri)
        
        # Test connection
        client.admin.command('ping')
        print("✅ Connection: SUCCESS")
        
        # Test database operations
        db = client.vehicle_insurance
        
        # Create a test collection and document
        test_collection = db["app_test"]
        
        # Insert sample insurance data
        sample_data = {
            "vehicle_type": "SUV",
            "insurance_type": "Comprehensive",
            "premium_amount": 5000,
            "customer_name": "Test Customer",
            "timestamp": "2024-01-01"
        }
        
        result = test_collection.insert_one(sample_data)
        print(f"✅ Data Insertion: SUCCESS (ID: {result.inserted_id})")
        
        # Read data back
        retrieved_data = test_collection.find_one({"_id": result.inserted_id})
        print(f"✅ Data Retrieval: SUCCESS")
        print(f"   Vehicle: {retrieved_data['vehicle_type']}")
        print(f"   Premium: ${retrieved_data['premium_amount']}")
        
        # Clean up
        test_collection.delete_one({"_id": result.inserted_id})
        print("✅ Data Cleanup: SUCCESS")
        
        # Show existing collections
        collections = db.list_collection_names()
        print(f"📂 Current collections: {collections}")
        
        client.close()
        print("\n🎉 ALL TESTS PASSED! MongoDB Atlas is ready for your application.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    final_test()