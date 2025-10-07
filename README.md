A comprehensive deployment guide for the Vehicle Insurance Prediction application on AWS EC2 with Python 3.11.

## 🚀 Quick Start Deployment

### Prerequisites
- AWS EC2 Ubuntu 24.04 instance
- SSH access to the instance
- Git repository access

### One-Command Deployment
Copy and execute the following script on your EC2 instance:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "🚀 Vehicle Insurance App Deployment"
echo "=========================================="

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and essential tools
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

# Create Python 3.11 virtual environment
echo "🔧 Setting up Python 3.11 virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
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

# Install additional packages
pip install pyyaml boto3 python-dotenv imbalanced-learn

# Create environment configuration
echo "🔐 Creating environment file..."
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=AKIA6ODU6E5Q7YS6A47K
AWS_SECRET_ACCESS_KEY=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
AWS_DEFAULT_REGION=us-east-1
MONGODB_URL=mongodb+srv://ayush210prasad_db_user:LgvjaRaelXiqE4a1@cluster0.dzwccg1.mongodb.net/
EOF

# Export environment variables
export $(cat .env | xargs)

# Start application server
echo "🎯 Starting application..."
nohup uvicorn app:app --host 0.0.0.0 --port 5000 > app.log 2>&1 &

# Wait for application initialization
echo "⏳ Waiting for application to start..."
sleep 10

# Verify deployment status
echo ""
echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "🌐 Access your application at: http://13.221.130.64:5000"
echo ""
echo "📊 Application Status:"
ps aux | grep uvicorn | grep -v grep
echo ""
echo "📋 Recent application logs:"
echo "=========================================="
tail -30 app.log
echo "=========================================="
📋 Post-Deployment Verification
Wait 10-15 seconds for installation and service initialization

Access the application: Open your browser to http://13.221.130.64:5000

Test functionality: Complete the form and click "Predict" to verify operation

🔧 Troubleshooting Guide
Diagnostic Commands
If you encounter issues after deployment:

bash
# Check application logs
cd ~/vehicle-inseurance
source venv/bin/activate
tail -50 app.log

# Test Python dependencies
python3.11 << 'PYEOF'
import sys
print(f"Python version: {sys.version}")

try:
    import fastapi
    print("✅ FastAPI - SUCCESS")
except Exception as e:
    print(f"❌ FastAPI - FAILED: {e}")

try:
    import pandas
    print("✅ Pandas - SUCCESS")
except Exception as e:
    print(f"❌ Pandas - FAILED: {e}")

try:
    import app
    print("✅ App module - SUCCESS")
except Exception as e:
    print(f"❌ App module - FAILED: {e}")
PYEOF
Application Restart Procedure
After system reboot, restart the application with:

bash
cd ~/vehicle-inseurance
source venv/bin/activate
export $(cat .env | xargs)
nohup uvicorn app:app --host 0.0.0.0 --port 5000 > app.log 2>&1 &
🛠️ Technical Features
Python 3.11 specifically installed (compatibility-optimized)

Exact package versioning from requirements

Dual HTTP method support (GET and POST on /)

PPA installation for Python 3.11 on Ubuntu 24.04

Version-conflict free package management

📁 Project Structure
text
~/vehicle-inseurance/
├── venv/                 # Python 3.11 virtual environment
├── app.py               # Main FastAPI application
├── .env                 # Environment configuration
├── app.log             # Application runtime logs
└── requirements.txt    # Python dependency specification
🔒 Security Configuration
Application service runs on port 5000

EC2 security group must allow inbound traffic on port 5000

Environment variables contain sensitive credentials

Production recommendation: Use AWS Secrets Manager for credential management

📞 Support & Maintenance
If you encounter deployment issues:

Check application logs: tail -f ~/vehicle-inseurance/app.log

Verify Python imports using the diagnostic script above

Confirm environment variables are properly configured

Note: This deployment specifically uses Python 3.11 to maintain compatibility with your local development environment and ensure package version stability.

text
