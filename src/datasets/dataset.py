import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import random
from skimage.transform import resize

class VolumeDataset(Dataset):
    """
    Loads entire 3D volumes (image and mask) and randomly samples patches.
    """
    def __init__(self, image_paths, mask_paths, patch_size=(64,64,64), num_patches_per_volume=10, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.augment = augment

        # Preload all volumes into memory to speed up training (optional)
        # If too big, you can load on-the-fly in __getitem__
        self.volumes = []
        for img_path, msk_path in zip(image_paths, mask_paths):
            image, mask = self.load_volume_pair(img_path, msk_path)
            self.volumes.append((image, mask))

    def __len__(self):
        # We consider each volume to provide multiple patches
        return len(self.volumes) * self.num_patches_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.num_patches_per_volume
        image, mask = self.volumes[vol_idx]

        # Sample a random patch
        img_patch, msk_patch = self.random_patch(image, mask)

        # Augmentation (optional)
        if self.augment:
            img_patch, msk_patch = self.augment_data(img_patch, msk_patch)

        # Add channel dimension
        img_patch = np.expand_dims(img_patch, axis=0)
        msk_patch = np.expand_dims(msk_patch, axis=0)

        # Convert to torch
        img_tensor = torch.from_numpy(img_patch).float()
        msk_tensor = torch.from_numpy(msk_patch).float()

        return img_tensor, msk_tensor

    def load_volume_pair(self, img_path, mask_path):
        img = nib.load(img_path)
        image_data = img.get_fdata()
        image_data = image_data.astype(np.float32)
        # Normalize
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)

        msk = nib.load(mask_path)
        mask_data = msk.get_fdata()
        mask_data = (mask_data > 0).astype(np.float32)

        # Optional: resize volumes to a certain size (e.g., if they are too large)
        # For simplicity, we use them as is, but consider resizing if needed.

        return image_data, mask_data

    def random_patch(self, image, mask):
        D, H, W = image.shape
        pd, ph, pw = self.patch_size

        # Random coordinates for the patch
        d_start = random.randint(0, max(D - pd, 0))
        h_start = random.randint(0, max(H - ph, 0))
        w_start = random.randint(0, max(W - pw, 0))

        img_patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        msk_patch = mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        # If the volume is smaller than the patch, we might need to pad
        # Check shapes and pad if necessary
        img_patch, msk_patch = self.pad_if_necessary(img_patch, msk_patch)

        return img_patch, msk_patch

    def pad_if_necessary(self, img_patch, msk_patch):
        desired = self.patch_size
        padded_img = np.zeros(desired, dtype=img_patch.dtype)
        padded_msk = np.zeros(desired, dtype=msk_patch.dtype)

        d, h, w = img_patch.shape
        pd, ph, pw = desired
        padded_img[:d, :h, :w] = img_patch
        padded_msk[:d, :h, :w] = msk_patch

        return padded_img, padded_msk

    def augment_data(self, img, msk):
        # Simple augmentation: random flip along one axis
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=2).copy()
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=1).copy()
        return img, msk
