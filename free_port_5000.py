import os
import subprocess
import sys

def free_port_5000():
    print("🔧 Freeing up port 5000...")
    
    try:
        # Run netstat to find processes using port 5000
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Look for port 5000 in listening state
        lines = result.stdout.split('\n')
        pids_to_kill = []
        
        for line in lines:
            if ':5000' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    pids_to_kill.append(pid)
                    print(f"📡 Found process {pid} using port 5000")
        
        if not pids_to_kill:
            print("✅ Port 5000 is free!")
            return True
        
        # Kill the processes
        for pid in pids_to_kill:
            try:
                subprocess.run(['taskkill', '/PID', pid, '/F'], check=True)
                print(f"✅ Killed process {pid}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Could not kill process {pid} (may require admin)")
        
        print("✅ Port 5000 should now be free!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    free_port_5000()