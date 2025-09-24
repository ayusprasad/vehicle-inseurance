import boto3

# Set your credentials
aws_access_key = "AKIA6ODU6E5QX5PYHHGM"
aws_secret_key = "nbYTQRZ+6kGj9jA+KRj79stXa/uWpieTpYFoiK5F"
bucket_name = "vehicle-insurance-models"

# Create S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name='us-east-1'
)

# Create bucket - FIXED: No LocationConstraint for us-east-1
try:
    # For us-east-1, don't specify LocationConstraint
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