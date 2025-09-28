import boto3
import os

# Get credentials from environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = "vehicle-insurance-models"

if not aws_access_key or not aws_secret_key:
    raise Exception("AWS credentials not found in environment variables.")

# Create S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name='us-east-1'
)

# Create bucket - FIXED: No LocationConstraint for us-east-1
try:
    s3.create_bucket(Bucket=bucket_name)
    print(f"Bucket {bucket_name} created successfully in us-east-1!")
    
    # Verify the bucket was created
    response = s3.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    if bucket_name in buckets:
        print(f"✅ Bucket '{bucket_name}' verified and ready!")
    else:
        print("❌ Bucket creation verification failed")
        
except Exception as e:
    print(f"Error creating bucket: {e}")