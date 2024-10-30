import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib  # For NIfTI files
import SimpleITK as sitk  # For DICOM files
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms

class MuscleDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Args:
            image_paths (list): List of paths to image files.
            mask_paths (list): List of paths to mask files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = self.load_image(self.image_paths[idx])
        mask = self.load_mask(self.mask_paths[idx])

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    def load_image_nifti(self, path):
        img = nib.load(path)
        img_data = img.get_fdata()
        # Normalize to [0, 1]
        img_data = img_data / np.max(img_data)
        # Convert to float32
        img_data = img_data.astype(np.float32)
        # Add channel dimension if needed
        img_data = np.expand_dims(img_data, axis=0)
        return img_data

    def load_image_dicom(self, path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        img_array = sitk.GetArrayFromImage(image)  # Shape: [slices, height, width]
        # For 2D images, select the appropriate slice
        img_data = img_array[0]  # Adjust index as needed
        # Normalize and convert to float32
        img_data = img_data / np.max(img_data)
        img_data = img_data.astype(np.float32)
        # Add channel dimension
        img_data = np.expand_dims(img_data, axis=0)
        return img_data

    def load_mask(self, path):
        # Similar loading logic for masks
        mask = nib.load(path)
        mask_data = mask.get_fdata()
        return mask_data

