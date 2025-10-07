Vehicle Insurance App - EC2 Deployment
A complete deployment guide for the Vehicle Insurance Prediction application on AWS EC2 with Python 3.11.

🚀 Quick Start Deployment
Prerequisites
AWS EC2 Ubuntu 24.04 instance

Access to instance via SSH

Git repository access

One-Command Deployment
Copy and paste the entire script below into your EC2 instance:

bash
#!/bin/bash
set -e

echo "=========================================="
echo "🚀 Vehicle Insurance App Deployment"
echo "=========================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and essentials
echo "🐍 Installing Python 3.11..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev git build-essential

# Clone repository
echo "📥 Cloning repository..."
cd ~
rm -rf vehicle-inseurance 2>/dev/null || true
git clone https://github.com/ayusprasad/vehicle-inseurance.git
cd vehicle-inseurance

# Create virtual environment with Python 3.11
echo "🔧 Setting up Python 3.11 virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install exact dependencies
echo "📚 Installing dependencies..."
pip install \
    fastapi==0.115.0 \
    uvicorn==0.30.3 \
    python-multipart==0.0.9 \
    jinja2==3.1.4 \
    pymongo==4.10.1 \
    dnspython==2.7.0 \
    numpy==2.1.2 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    dill==0.3.9 \
    from-root==1.3.0 \
    certifi==2024.8.30 \
    python-dateutil==2.9.0.post0 \
    pytz==2025.1 \
    tzdata==2025.1

# Install additional required packages
pip install pyyaml boto3 python-dotenv imbalanced-learn

# Create .env file
echo "🔐 Creating environment file..."
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=AKIA6ODU6E5Q7YS6A47K
AWS_SECRET_ACCESS_KEY=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
AWS_DEFAULT_REGION=us-east-1
MONGODB_URL=mongodb+srv://ayush210prasad_db_user:LgvjaRaelXiqE4a1@cluster0.dzwccg1.mongodb.net/
EOF

# Export environment variables
export $(cat .env | xargs)

# Start application
echo "🎯 Starting application..."
nohup uvicorn app:app --host 0.0.0.0 --port 5000 > app.log 2>&1 &

# Wait for startup
echo "⏳ Waiting for application to start..."
sleep 10

# Check status
echo ""
echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "🌐 Access your app at: http://13.221.130.64:5000"
echo ""
echo "📊 Application Status:"
ps aux | grep uvicorn | grep -v grep
echo ""
echo "📋 Last 30 log lines:"
echo "=========================================="
tail -30 app.log
echo "=========================================="
echo ""
echo "📝 Useful Commands:"
echo "  View logs: tail -f ~/vehicle-inseurance/app.log"
echo "  Check status: ps aux | grep uvicorn"
echo "  Stop app: pkill -f uvicorn"
echo ""
echo "🔄 To restart:"
echo "  cd ~/vehicle-inseurance"
echo "  source venv/bin/activate"
echo "  export \$(cat .env | xargs)"
echo "  nohup uvicorn app:app --host 0.0.0.0 --port 5000 > app.log 2>&1 &"
echo ""
📋 After Running the Script
Wait 10-15 seconds for everything to install and start

Open your browser: http://13.221.130.64:5000

Test the form - fill it out and click "Predict"

🔧 Troubleshooting
If You See ANY Errors After Running
bash
# Check what's happening
cd ~/vehicle-inseurance
source venv/bin/activate
tail -50 app.log

# Test Python imports
python3.11 << 'PYEOF'
import sys
print(f"Python version: {sys.version}")

try:
    import fastapi
    print("✅ FastAPI")
except Exception as e:
    print(f"❌ FastAPI: {e}")

try:
    import pandas
    print("✅ Pandas")
except Exception as e:
    print(f"❌ Pandas: {e}")

try:
    import app
    print("✅ App module")
except Exception as e:
    print(f"❌ App: {e}")
PYEOF
To Restart After Reboot
bash
cd ~/vehicle-inseurance
source venv/bin/activate
export $(cat .env | xargs)
nohup uvicorn app:app --host 0.0.0.0 --port 5000 > app.log 2>&1 &
🛠️ Key Features
Python 3.11 specifically installed (not 3.12)

Exact package versions from requirements

POST method support (app supports both GET and POST on /)

PPA installation for Python 3.11 on Ubuntu 24.04

No version conflicts - all packages compatible with Python 3.11

📁 Project Structure
text
~/vehicle-inseurance/
├── venv/                 # Python 3.11 virtual environment
├── app.py               # Main FastAPI application
├── .env                 # Environment variables
├── app.log             # Application logs
└── requirements.txt    # Python dependencies
🔒 Security Notes
The application runs on port 5000

Ensure EC2 security group allows inbound traffic on port 5000

Environment variables contain sensitive credentials

Consider using AWS Secrets Manager for production

📞 Support
If you encounter any issues:

Check the application logs: tail -f ~/vehicle-inseurance/app.log

Verify Python imports using the troubleshooting script above

Ensure all environment variables are properly set

Note: This deployment uses Python 3.11 specifically to match your local development environment and ensure compatibility with all package versions.