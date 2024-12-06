# datasets/dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pydicom
from skimage.transform import resize
from scipy.ndimage import zoom

class SliceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, desired_size=(256, 256), target_spacing=(3.0, 1.7188, 1.7188)):
        self.image_slices = []
        self.mask_slices = []
        self.desired_size = desired_size
        self.target_spacing = target_spacing  # (Z, Y, X)

        for img_path, mask_path in zip(image_paths, mask_paths):
            image_slices = self.load_slices(img_path, is_mask=False)
            mask_slices = self.load_slices(mask_path, is_mask=True)
            assert len(image_slices) == len(mask_slices), f"Mismatch in slices at {img_path}"

            self.image_slices.extend(image_slices)
            self.mask_slices.extend(mask_slices)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        # Determine neighboring slice indices
        idx_prev = max(idx - 1, 0)
        idx_next = min(idx + 1, len(self.image_slices) - 1)

        # Get slices
        image_prev = self.image_slices[idx_prev]
        image_current = self.image_slices[idx]
        image_next = self.image_slices[idx_next]

        # Resize each slice before stacking
        image_prev = resize(image_prev, self.desired_size, mode='reflect', anti_aliasing=True)
        image_current = resize(image_current, self.desired_size, mode='reflect', anti_aliasing=True)
        image_next = resize(image_next, self.desired_size, mode='reflect', anti_aliasing=True)

        # Stack to create 3-channel image
        image_3ch = np.stack([image_prev, image_current, image_next], axis=0)  # (3, H, W)

        # Get and resize mask
        mask = self.mask_slices[idx]
        mask = resize(mask, self.desired_size, order=0, preserve_range=True, anti_aliasing=False)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_3ch).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return image_tensor, mask_tensor

    def load_slices(self, path, is_mask=False):
        if os.path.isdir(path):
            # DICOM data
            slices = self.load_dicom_slices(path)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            # NIfTI data
            slices = self.load_nifti_slices(path, is_mask=is_mask)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return slices

    def load_dicom_slices(self, dicom_dir):
        dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        dicom_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
        slices = []
        for file in dicom_files:
            ds = pydicom.dcmread(file)
            image = ds.pixel_array.astype(np.float32)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            slices.append(image)
        return slices

    def load_nifti_slices(self, nifti_path, is_mask=False):
        img = nib.load(nifti_path)
        data = img.get_fdata()
        header = img.header

        # Original voxel spacing (X, Y, Z)
        voxel_spacing = header.get_zooms()
        # Reorder to (Z, Y, X)
        voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])

        if is_mask:
            data = data.astype(np.float32)
            data = (data > 0).astype(np.float32)
            order = 0
        else:
            data = data.astype(np.float32)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            order = 1

        # Resample volume to target_spacing
        zoom_factors = (
            voxel_spacing[0] / self.target_spacing[0],
            voxel_spacing[1] / self.target_spacing[1],
            voxel_spacing[2] / self.target_spacing[2]
        )
        data_resampled = zoom(data, zoom_factors, order=order)

        # Extract slices along Z-axis
        slices = [data_resampled[:, :, i] for i in range(data_resampled.shape[2])]
        return slices
