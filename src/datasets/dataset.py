import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from skimage.transform import resize
import random

class VolumeDataset(Dataset):
    def __init__(
        self,
        water_paths,
        fat_paths,
        mask_paths,
        desired_size=(32, 128, 128),
        target_spacing=(1.75, 1.75, 3.0),
        augment=True
    ):
        """
        :param water_paths: list of water image paths
        :param fat_paths:   list of fat image paths
        :param mask_paths:  list of mask image paths
        :param desired_size: final (D,H,W) after resizing
        :param target_spacing: the target voxel spacing to resample to
        :param augment: whether to apply data augmentation
        """
        self.water_paths = water_paths
        self.fat_paths   = fat_paths
        self.mask_paths  = mask_paths
        self.desired_size = desired_size
        self.target_spacing = target_spacing
        self.augment = augment

    def __len__(self):
        return len(self.water_paths)

    def __getitem__(self, idx):
        water_path = self.water_paths[idx]
        fat_path   = self.fat_paths[idx]
        mask_path  = self.mask_paths[idx]

        water, fat, mask = self.load_and_preprocess(water_path, fat_path, mask_path)

        if self.augment:
            water, fat, mask = self.random_augmentations(water, fat, mask)

        # Return them as (D,H,W) for each.
        # Collation in DataLoader will stack them into (B,D,H,W).
        water_tensor = torch.from_numpy(water).float()
        fat_tensor   = torch.from_numpy(fat).float()
        mask_tensor  = torch.from_numpy(mask).float()

        return water_tensor, fat_tensor, mask_tensor

    def load_and_preprocess(self, water_path, fat_path, mask_path):
        # ---------------------
        # Load water
        # ---------------------
        w_img = nib.load(water_path)
        w_data = w_img.get_fdata().astype(np.float32)
        w_header = w_img.header
        w_spacing = w_header.get_zooms()  # (X_spacing, Y_spacing, Z_spacing)

        # Simple intensity normalization
        w_99 = np.percentile(w_data, 99)
        w_data = np.clip(w_data, 0, w_99) / (w_99 + 1e-8)

        # Resample water
        w_zoom_factors = (
            w_spacing[0] / self.target_spacing[0],
            w_spacing[1] / self.target_spacing[1],
            w_spacing[2] / self.target_spacing[2]
        )
        w_data_resampled = zoom(w_data, w_zoom_factors, order=1)
        # Reorder to (D,H,W)
        w_data_resampled = np.transpose(w_data_resampled, (2, 0, 1))
        # Resize
        w_data_resized = resize(
            w_data_resampled,
            self.desired_size,
            mode='reflect',
            anti_aliasing=True
        )

        # ---------------------
        # Load fat
        # ---------------------
        f_img = nib.load(fat_path)
        f_data = f_img.get_fdata().astype(np.float32)
        f_header = f_img.header
        f_spacing = f_header.get_zooms()

        f_99 = np.percentile(f_data, 99)
        f_data = np.clip(f_data, 0, f_99) / (f_99 + 1e-8)

        f_zoom_factors = (
            f_spacing[0] / self.target_spacing[0],
            f_spacing[1] / self.target_spacing[1],
            f_spacing[2] / self.target_spacing[2]
        )
        f_data_resampled = zoom(f_data, f_zoom_factors, order=1)
        f_data_resampled = np.transpose(f_data_resampled, (2, 0, 1))
        f_data_resized = resize(
            f_data_resampled,
            self.desired_size,
            mode='reflect',
            anti_aliasing=True
        )

        # ---------------------
        # Load mask
        # ---------------------
        m_img = nib.load(mask_path)
        m_data = m_img.get_fdata().astype(np.float32)
        # Binarize
        m_data = (m_data > 0).astype(np.float32)
        m_header = m_img.header
        m_spacing = m_header.get_zooms()

        m_zoom_factors = (
            m_spacing[0] / self.target_spacing[0],
            m_spacing[1] / self.target_spacing[1],
            m_spacing[2] / self.target_spacing[2]
        )
        m_data_resampled = zoom(m_data, m_zoom_factors, order=0)
        m_data_resampled = np.transpose(m_data_resampled, (2, 0, 1))
        m_data_resized = resize(
            m_data_resampled,
            self.desired_size,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.float32)

        return w_data_resized, f_data_resized, m_data_resized

    def random_augmentations(self, water, fat, mask):
        """
        A few random augmentations for 3D data.
        All volumes are shape (D,H,W) at this point.
        We apply the same transformations to water, fat, and mask
        to keep them aligned.
        """
        # 1) Gaussian blur
        if random.random() < 0.5:
            sigma = random.uniform(0, 1.0)  # small blur
            water = gaussian_filter(water, sigma=sigma)
            fat   = gaussian_filter(fat, sigma=sigma)
            # mask is not blurred (since it's binary), but
            # if you want to slightly blur it, do with order=0
            # but typically we do not blur the mask.

        # 2) Intensity scaling
        if random.random() < 0.5:
            scale_factor_w = random.uniform(0.9, 1.1)
            scale_factor_f = random.uniform(0.9, 1.1)
            water = water * scale_factor_w
            fat   = fat * scale_factor_f
            water = np.clip(water, 0, 1.0)
            fat   = np.clip(fat,   0, 1.0)

        # 3) Random shift in D,H,W
        if random.random() < 0.5:
            max_shift = 5  # voxels
            shift_d = random.randint(-max_shift, max_shift)
            shift_h = random.randint(-max_shift, max_shift)
            shift_w = random.randint(-max_shift, max_shift)

            water = np.roll(water, shift=shift_d, axis=0)
            water = np.roll(water, shift=shift_h, axis=1)
            water = np.roll(water, shift=shift_w, axis=2)

            fat   = np.roll(fat, shift=shift_d, axis=0)
            fat   = np.roll(fat, shift=shift_h, axis=1)
            fat   = np.roll(fat, shift=shift_w, axis=2)

            mask  = np.roll(mask, shift=shift_d, axis=0)
            mask  = np.roll(mask, shift=shift_h, axis=1)
            mask  = np.roll(mask, shift=shift_w, axis=2)

        return water, fat, mask
