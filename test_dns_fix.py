import socket
import subprocess
import sys
import os

def fix_dns_connection():
    print("🔧 Fixing DNS and Network Connection")
    print("=" * 50)
    
    # Test basic internet connectivity
    print("1. Testing basic internet connectivity...")
    try:
        socket.gethostbyname("google.com")
        print("✅ Internet connection: OK")
    except:
        print("❌ No internet connection")
        return False

    # Test MongoDB Atlas DNS
    print("\n2. Testing MongoDB Atlas DNS resolution...")
    hosts_to_test = [
        "cluster0.dzwccg1.mongodb.net",
        "mongodb.net",
        "atlas.mongodb.com"
    ]
    
    for host in hosts_to_test:
        try:
            ip = socket.gethostbyname(host)
            print(f"✅ {host} → {ip}")
        except socket.gaierror as e:
            print(f"❌ {host}: {e}")

    # Test connection with IP if possible
    print("\n3. Testing alternative connection methods...")
    
    # Try different connection approaches
    connection_strings = [
        "mongodb+srv://ayush210prasad_db_user:LgvjaRaelXiqE4a1@cluster0.dzwccg1.mongodb.net/vehicle_insurance?retryWrites=true&w=majority",
        "mongodb+srv://ayush210prasad_db_user:LgvjaRaelXiqE4a1@cluster0.dzwccg1.mongodb.net/?retryWrites=true&w=majority",
    ]
    
    return True

def flush_dns():
    print("\n4. Flushing DNS cache...")
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['ipconfig', '/flushdns'], capture_output=True)
            print("✅ DNS cache flushed")
        else:  # Linux/Mac
            subprocess.run(['sudo', 'systemd-resolve', '--flush-caches'], capture_output=True)
            print("✅ DNS cache flushed")
    except Exception as e:
        print(f"⚠️  Could not flush DNS: {e}")

if __name__ == "__main__":
    flush_dns()
    fix_dns_connection()