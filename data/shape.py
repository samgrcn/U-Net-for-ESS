import os
import nibabel as nib
import numpy as np

# Root directory
root_dir = "test_belgium_data/"

# Target file patterns
target_files = [' mDIXON-Quant_BH_v3.nii', ' mDIXON-Quant_BH.nii']

# To store all shapes
shapes = []

# Traverse directories
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file in target_files:
            file_path = os.path.join(subdir, file)
            img = nib.load(file_path)  # Load NIfTI file
            data_shape = img.shape  # Extract shape
            shapes.append(data_shape)
            print(f"File: {file_path}, Shape: {data_shape}")

# Calculate the average shape
if shapes:
    avg_shape = np.mean(np.array(shapes), axis=0)
    print(f"\nAverage Shape of Dataset: {avg_shape}")
else:
    print("No target files found.")
