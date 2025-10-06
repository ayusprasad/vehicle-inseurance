import socket
import requests

def network_test():
    print("🌐 Network Diagnostic Test")
    print("=" * 40)
    
    tests = [
        ("Google DNS", "8.8.8.8"),
        ("Google", "google.com"),
        ("MongoDB Atlas", "cluster0.dzwccg1.mongodb.net"),
        ("MongoDB", "mongodb.com")
    ]
    
    for name, host in tests:
        try:
            if host.replace('.', '').isdigit():  # IP address
                socket.create_connection((host, 80), timeout=5)
                print(f"✅ {name} ({host}): Reachable")
            else:  # Hostname
                ip = socket.gethostbyname(host)
                socket.create_connection((ip, 80), timeout=5)
                print(f"✅ {name} ({host} → {ip}): Reachable")
        except Exception as e:
            print(f"❌ {name} ({host}): {e}")

if __name__ == "__main__":
    network_test()