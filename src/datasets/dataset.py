# datasets/dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import zoom
from utils.utils import normalize

class SliceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, desired_size=(256, 256), target_spacing=(3.0, 1.7188, 1.7188)):
        self.image_slices = []
        self.mask_slices = []
        self.desired_size = desired_size
        self.target_spacing = target_spacing  # (Z, Y, X)

        for img_path, mask_path in zip(image_paths, mask_paths):
            image_slices = self.load_nifti_slices(img_path, is_mask=False)
            mask_slices = self.load_nifti_slices(mask_path, is_mask=True)
            assert len(image_slices) == len(mask_slices), f"Mismatch in number of slices between image and mask at {img_path}"

            self.image_slices.extend(image_slices)
            self.mask_slices.extend(mask_slices)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        # Get indices for neighboring slices
        idx_prev = max(idx - 1, 0)
        idx_next = min(idx + 1, len(self.image_slices) - 1)

        # Get the three slices
        image_prev = self.image_slices[idx_prev]
        image_current = self.image_slices[idx]
        image_next = self.image_slices[idx_next]

        # Stack them to create a 3-channel image
        image = np.stack([image_prev, image_current, image_next], axis=0)  # Shape: (3, H, W)

        # Get the mask for the current slice
        mask = self.mask_slices[idx]

        # Resize image and mask to desired_size
        resized_image = np.zeros((3, *self.desired_size), dtype=image.dtype)
        for c in range(3):
            resized_image[c] = resize(
                image[c],
                self.desired_size,
                mode='reflect',
                anti_aliasing=True
            )

        mask = resize(
            mask,
            self.desired_size,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )

        # Convert to tensors
        image = torch.from_numpy(resized_image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension

        return image, mask

    def load_nifti_slices(self, nifti_path, is_mask=False):
        img = nib.load(nifti_path)
        data = img.get_fdata()
        header = img.header
        affine = img.affine

        # Get voxel spacing
        voxel_spacing = header.get_zooms()  # (X, Y, Z)
        voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])  # Convert to (Z, Y, X)

        if is_mask:
            data = data.astype(np.float32)
            data = (data > 0).astype(np.float32)
        else:
            data = data.astype(np.float32)
            data = normalize(data)

        # Resample volume to target_spacing
        zoom_factors = (
            voxel_spacing[0] / self.target_spacing[0],
            voxel_spacing[1] / self.target_spacing[1],
            voxel_spacing[2] / self.target_spacing[2]
        )
        data_resampled = zoom(data, zoom_factors, order=1 if not is_mask else 0)

        # Extract slices along Z-axis
        slices = [data_resampled[:, :, i] for i in range(data_resampled.shape[2])]
        return slices
