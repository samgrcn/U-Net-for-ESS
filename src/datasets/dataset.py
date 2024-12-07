import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import random
from scipy.ndimage import zoom

class VolumeDataset(Dataset):
    """
    Loads entire 3D volumes, resamples them to target spacing, and extracts random patches.
    """

    def __init__(self, image_paths, mask_paths, patch_size=(64,64,64), num_patches_per_volume=10, augment=False, target_spacing=(3.0, 1.7188, 1.7188)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.augment = augment
        self.target_spacing = target_spacing

        self.volumes = []
        for img_path, msk_path in zip(image_paths, mask_paths):
            image, mask = self.load_volume_pair(img_path, msk_path)
            self.volumes.append((image, mask))

    def __len__(self):
        return len(self.volumes) * self.num_patches_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.num_patches_per_volume
        image, mask = self.volumes[vol_idx]
        img_patch, msk_patch = self.random_patch(image, mask)

        if self.augment:
            img_patch, msk_patch = self.augment_data(img_patch, msk_patch)

        img_patch = np.expand_dims(img_patch, axis=0)
        msk_patch = np.expand_dims(msk_patch, axis=0)

        img_tensor = torch.from_numpy(img_patch).float()
        msk_tensor = torch.from_numpy(msk_patch).float()

        return img_tensor, msk_tensor

    def load_volume_pair(self, img_path, mask_path):
        img = nib.load(img_path)
        image_data = img.get_fdata().astype(np.float32)
        msk = nib.load(mask_path)
        mask_data = msk.get_fdata().astype(np.float32)

        # Binarize mask
        mask_data = (mask_data > 0).astype(np.float32)

        # Normalize image
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)

        # Get original spacing
        original_spacing = img.header.get_zooms()  # (Xsp, Ysp, Zsp)
        # Resample to target spacing
        image_data = self.resample_volume(image_data, original_spacing, self.target_spacing, order=1)
        mask_data = self.resample_volume(mask_data, original_spacing, self.target_spacing, order=0)

        return image_data, mask_data

    def resample_volume(self, data, original_spacing, target_spacing, order=1):
        # original_spacing and target_spacing are (Xsp, Ysp, Zsp)
        # data.shape corresponds to (X, Y, Z)
        zoom_factors = (original_spacing[0]/target_spacing[0],
                        original_spacing[1]/target_spacing[1],
                        original_spacing[2]/target_spacing[2])
        # Use scipy.ndimage.zoom
        resampled = zoom(data, zoom_factors, order=order)
        return resampled

    def random_patch(self, image, mask):
        D, H, W = image.shape
        pd, ph, pw = self.patch_size

        d_start = random.randint(0, max(D - pd, 0))
        h_start = random.randint(0, max(H - ph, 0))
        w_start = random.randint(0, max(W - pw, 0))

        img_patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        msk_patch = mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        # Pad if necessary
        img_patch, msk_patch = self.pad_if_necessary(img_patch, msk_patch)
        return img_patch, msk_patch

    def pad_if_necessary(self, img_patch, msk_patch):
        desired = self.patch_size
        d, h, w = img_patch.shape
        pd, ph, pw = desired
        padded_img = np.zeros(desired, dtype=img_patch.dtype)
        padded_msk = np.zeros(desired, dtype=msk_patch.dtype)

        padded_img[:d, :h, :w] = img_patch
        padded_msk[:d, :h, :w] = msk_patch

        return padded_img, padded_msk

    def augment_data(self, img, msk):
        # Simple random flips
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=2).copy()
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=1).copy()
        return img, msk
