import pymongo
import sys

def test_direct_connection():
    print("🚀 Attempting Direct MongoDB Connection")
    print("=" * 50)
    
    # Method 1: Direct connection string
    uri = "mongodb+srv://ayush210prasad_db_user:LgvjaRaelXiqE4a1@cluster0.dzwccg1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    print("Testing connection...")
    
    try:
        client = pymongo.MongoClient(
            uri,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        
        # Test connection
        client.admin.command('ping')
        print("✅ SUCCESS: Connected to MongoDB Atlas!")
        
        # Show database info
        dbs = client.list_database_names()
        print(f"📊 Databases: {dbs}")
        
        return True
        
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(f"❌ Timeout error: {e}")
        print("\n💡 This usually means:")
        print("   - DNS resolution failed")
        print("   - Network firewall blocking connection")
        print("   - Internet service provider issues")
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    return False

if __name__ == "__main__":
    test_direct_connection()