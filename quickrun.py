# Run this in your Python interpreter to see where the preprocessor actually is
import os
artifact_dir = "artifact"
latest_dir = "09_24_2025_19_47_36"  # Use your latest directory

# Check data_transformation directory structure
dt_path = os.path.join(artifact_dir, latest_dir, "data_transformation")
if os.path.exists(dt_path):
    print("Data Transformation directory contents:")
    for root, dirs, files in os.walk(dt_path):
        level = root.replace(dt_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
else:
    print(f"Data transformation directory not found: {dt_path}")