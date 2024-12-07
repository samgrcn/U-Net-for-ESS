import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from src.models.unet3d import UNet3D
from skimage.transform import resize

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_test_data_paths(test_dir, has_masks=True):
    patient_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    image_paths = []
    mask_paths = []
    for patient_dir in patient_dirs:
        files_in_patient = os.listdir(patient_dir)
        image_file = None
        for fname in [' mDIXON-Quant_BH_v3.nii', ' mDIXON-Quant_BH.nii']:
            if fname in files_in_patient:
                image_file = fname
                break
        if image_file is None:
            print(f"No image file found in {patient_dir}")
            continue
        image_path = os.path.join(patient_dir, image_file)
        image_paths.append(image_path)
        if has_masks:
            mask_file = None
            for fname in ['erector.nii', 'erector.nii.gz']:
                if fname in files_in_patient:
                    mask_file = fname
                    break
            if mask_file is None:
                print(f"No mask file found in {patient_dir}")
                continue
            mask_path = os.path.join(patient_dir, mask_file)
            if not os.path.exists(mask_path):
                print(f"No mask file found in {patient_dir}")
                continue
            mask_paths.append(mask_path)
    if has_masks:
        return image_paths, mask_paths
    else:
        return image_paths

def load_nifti_image(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine

    data = data.astype(np.float32)
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    return data, affine

def load_nifti_mask(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    data = (data > 0).astype(np.uint8)
    return data

def load_model(checkpoint_path, device):
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def predict_volume(model, volume, patch_size=(64,64,64), threshold=0.5):
    # Sliding window inference if needed
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    pred = np.zeros((D,H,W), dtype=np.float32)
    count_map = np.zeros((D,H,W), dtype=np.float32)

    # Steps
    dz = max(pd//2, 1)
    dy = max(ph//2, 1)
    dx = max(pw//2, 1)

    for d_start in range(0, D, dz):
        for h_start in range(0, H, dy):
            for w_start in range(0, W, dx):
                d_end = min(d_start+pd, D)
                h_end = min(h_start+ph, H)
                w_end = min(w_start+pw, W)

                patch = volume[d_start:d_end, h_start:h_end, w_start:w_end]

                # Pad if needed
                patch_padded = np.zeros(patch_size, dtype=np.float32)
                dd, hh, ww = patch.shape
                patch_padded[:dd, :hh, :ww] = patch
                patch_tensor = torch.from_numpy(patch_padded[None,None,...]).float().to(DEVICE)

                with torch.no_grad():
                    out = model(patch_tensor)
                prob = torch.sigmoid(out).cpu().numpy()[0,0]

                # Unpad prob
                prob = prob[:dd,:hh,:ww]

                pred[d_start:d_start+dd, h_start:h_start+hh, w_start:w_start+ww] += prob
                count_map[d_start:d_start+dd, h_start:h_start+hh, w_start:w_start+ww] += 1

    pred = pred / (count_map + 1e-8)
    pred_mask = (pred > threshold).astype(np.uint8)
    return pred_mask

def main():
    CHECKPOINT_PATH = './outputs/checkpoints/3D-Unet/best_model.pth.tar'
    TEST_PARIS_DIR = './data/test_paris_data/'
    TEST_BELGIUM_DIR = './data/test_belgium_data/'
    OUTPUT_DIR = './outputs/predictions/3D-Unet/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, DEVICE)

    paris_image_paths, paris_mask_paths = get_test_data_paths(TEST_PARIS_DIR, True)
    belgium_image_paths, belgium_mask_paths = get_test_data_paths(TEST_BELGIUM_DIR, True)

    # Take first 4
    paris_image_paths = paris_image_paths[:4]
    paris_mask_paths = paris_mask_paths[:4]
    belgium_image_paths = belgium_image_paths[:4]
    belgium_mask_paths = belgium_mask_paths[:4]

    paris_slices = []
    belgium_slices = []

    for idx, (image_path, mask_path) in enumerate(zip(paris_image_paths, paris_mask_paths)):
        image_data, affine = load_nifti_image(image_path)
        mask_data = load_nifti_mask(mask_path)
        pred_mask = predict_volume(model, image_data, patch_size=(64,64,64), threshold=0.5)

        # Save prediction
        pred_nifti = nib.Nifti1Image(pred_mask.astype(np.float32), affine)
        output_mask_path = os.path.join(OUTPUT_DIR, f'paris_patient_{idx+1}_pred_mask.nii.gz')
        nib.save(pred_nifti, output_mask_path)

        # Middle slice
        mid = image_data.shape[2]//2
        paris_slices.append((image_data[:,:,mid], mask_data[:,:,mid], pred_mask[:,:,mid]))

    for idx, (image_path, mask_path) in enumerate(zip(belgium_image_paths, belgium_mask_paths)):
        image_data, affine = load_nifti_image(image_path)
        mask_data = load_nifti_mask(mask_path)
        pred_mask = predict_volume(model, image_data, patch_size=(64,64,64), threshold=0.5)

        pred_nifti = nib.Nifti1Image(pred_mask.astype(np.float32), affine)
        output_mask_path = os.path.join(OUTPUT_DIR, f'belgium_patient_{idx+1}_pred_mask.nii.gz')
        nib.save(pred_nifti, output_mask_path)

        # Middle slice
        mid = image_data.shape[2]//2
        belgium_slices.append((image_data[:,:,mid], mask_data[:,:,mid], pred_mask[:,:,mid]))

    # Plot results
    def plot_slices(slices, title_prefix, output_name):
        num = len(slices)
        fig, axes = plt.subplots(num, 4, figsize=(20, 5*num))
        if num == 1:
            axes = [axes]  # handle the single-row case
        for i, (img_slice, gt_slice, pred_slice) in enumerate(slices):
            axes[i][0].imshow(img_slice, cmap='gray')
            axes[i][0].set_title(f'{title_prefix} Patient {i+1} - Image')
            axes[i][0].axis('off')

            axes[i][1].imshow(gt_slice, cmap='gray')
            axes[i][1].set_title('Ground Truth')
            axes[i][1].axis('off')

            axes[i][2].imshow(pred_slice, cmap='gray')
            axes[i][2].set_title('Predicted')
            axes[i][2].axis('off')

            overlay = np.zeros((*img_slice.shape,3))
            overlay[(gt_slice==1)&(pred_slice==1)] = [0,1,0]
            overlay[(gt_slice==1)&(pred_slice==0)] = [0,0,1]
            overlay[(gt_slice==0)&(pred_slice==1)] = [1,0,0]

            axes[i][3].imshow(img_slice, cmap='gray')
            axes[i][3].imshow(overlay, alpha=0.3)
            axes[i][3].set_title('Overlay')
            axes[i][3].axis('off')
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, output_name)
        plt.savefig(path)
        plt.close(fig)

    plot_slices(paris_slices, 'Paris', 'paris_predictions.png')
    plot_slices(belgium_slices, 'Belgium', 'belgium_predictions.png')

if __name__ == '__main__':
    main()
