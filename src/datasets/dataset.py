# dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pydicom
from skimage.transform import resize

class SliceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, desired_size=(256, 256)):
        self.image_slices = []
        self.mask_slices = []
        self.slice_indices = []
        self.volume_start_indices = []
        self.desired_size = desired_size

        current_index = 0
        for img_path, mask_path in zip(image_paths, mask_paths):
            image_slices = self.load_slices(img_path, is_mask=False)
            mask_slices = self.load_slices(mask_path, is_mask=True)
            assert len(image_slices) == len(mask_slices), f"Mismatch in number of slices between image and mask at {img_path}"
            # Append slices to the dataset lists
            self.image_slices.extend(image_slices)
            self.mask_slices.extend(mask_slices)
            # Record the indices
            num_slices = len(image_slices)
            self.slice_indices.extend([(current_index + i) for i in range(num_slices)])
            self.volume_start_indices.append(current_index)
            current_index += num_slices

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        mask = self.mask_slices[idx]

        # Get previous and next slices if available, handle volume boundaries
        prev_idx = idx - 1 if idx - 1 >= 0 and (idx - 1) in self.slice_indices else idx
        next_idx = idx + 1 if idx + 1 < len(self.image_slices) and (idx + 1) in self.slice_indices else idx

        # Check for edge cases
        if idx in self.volume_start_indices:
            prev_idx = idx
        if (idx + 1) in self.volume_start_indices:
            next_idx = idx

        prev_image = self.image_slices[prev_idx]
        next_image = self.image_slices[next_idx]

        # Stack slices
        image = np.stack([prev_image, image, next_image], axis=0)

        # Resize image and mask
        image = resize(
            image,
            (3, self.desired_size[0], self.desired_size[1]),
            mode='reflect',
            anti_aliasing=True
        )
        mask = resize(
            mask,
            (self.desired_size[0], self.desired_size[1]),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

    def load_slices(self, path, is_mask=False):
        if os.path.isdir(path):
            # DICOM slices
            slices = self.load_dicom_slices(path)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            # Load NIfTI slices
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
            # Normalize image to [0, 1]
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            slices.append(image)
        return slices

    def load_nifti_slices(self, nifti_path, is_mask=False):
        img = nib.load(nifti_path)
        data = img.get_fdata()

        if is_mask:
            data = data.astype(np.float32)
            # Binarize the mask if necessary
            data = (data > 0).astype(np.float32)
        else:
            data = data.astype(np.float32)
            # Normalize image data to [0, 1]
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Extract slices along Z-axis
        slices = [data[:, :, i] for i in range(data.shape[2])]
        return slices
