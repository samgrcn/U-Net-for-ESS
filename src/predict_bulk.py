import os
import torch
import numpy as np
import nibabel as nib
from src.models.unet3d import UNet3D
from skimage.transform import resize
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_bulk_image_paths(test_dir):
    patient_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir)
                    if os.path.isdir(os.path.join(test_dir, d))]
    image_paths = []
    for p_dir in patient_dirs:
        files_in_patient = os.listdir(p_dir)
        water_image = None
        for f in files_in_patient:
            if 'water' in f and (f.endswith('.nii') or f.endswith('.nii.gz')):
                water_image = f
                break
        if water_image is None:
            print(f"No 'water' image found in {p_dir}. Skipping.")
            continue
        image_path = os.path.join(p_dir, water_image)
        patient_name = os.path.basename(p_dir)
        image_paths.append((patient_name, image_path))
    return image_paths

def load_nifti_image(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    data = (data - data.min())/(data.max()-data.min()+1e-8)
    affine = img.affine
    return data, affine

def load_model(checkpoint_path, device):
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def predict_volume(model, volume, patch_size=(64,64,64), threshold=0.5):
    # Same sliding window approach as before
    D,H,W = volume.shape
    pd,ph,pw = patch_size
    pred = np.zeros((D,H,W), dtype=np.float32)
    count_map = np.zeros((D,H,W), dtype=np.float32)

    dz = max(pd//2,1)
    dy = max(ph//2,1)
    dx = max(pw//2,1)

    for d_start in range(0,D,dz):
        for h_start in range(0,H,dy):
            for w_start in range(0,W,dx):
                d_end = min(d_start+pd,D)
                h_end = min(h_start+ph,H)
                w_end = min(w_start+pw,W)

                patch = volume[d_start:d_end,h_start:h_end,w_start:w_end]
                patch_padded = np.zeros(patch_size, dtype=np.float32)
                dd,hh,ww = patch.shape
                patch_padded[:dd,:hh,:ww] = patch
                patch_tensor = torch.from_numpy(patch_padded[None,None,...]).float().to(DEVICE)

                with torch.no_grad():
                    out = model(patch_tensor)
                prob = torch.sigmoid(out).cpu().numpy()[0,0]

                prob = prob[:dd,:hh,:ww]

                pred[d_start:d_start+dd,h_start:h_start+hh,w_start:w_start+ww] += prob
                count_map[d_start:d_start+dd,h_start:h_start+hh,w_start:w_start+ww] += 1

    pred = pred / (count_map+1e-8)
    pred_mask = (pred>threshold).astype(np.uint8)
    return pred_mask

def main():
    CHECKPOINT_PATH = './outputs/checkpoints/3D-Unet/best_model.pth.tar'
    TEST_BULK_DIR = './data/test_belgium_bulk/'
    OUTPUT_DIR = './outputs/predictions/3D-Unet/bulk/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, DEVICE)
    bulk_images = get_bulk_image_paths(TEST_BULK_DIR)

    images_list = []
    preds_list = []
    names_list = []

    for patient_name, image_path in bulk_images:
        print(f"Predicting for {patient_name}")
        image_data, affine = load_nifti_image(image_path)
        pred_mask = predict_volume(model, image_data, patch_size=(64,64,64), threshold=0.5)

        # Save
        output_path = os.path.join(OUTPUT_DIR, f'{patient_name}_pred_mask.nii.gz')
        nib.save(nib.Nifti1Image(pred_mask.astype(np.float32), affine), output_path)
        print(f"Saved prediction at {output_path}")

        images_list.append(image_data)
        preds_list.append(pred_mask)
        names_list.append(patient_name)

    # Plot all 15 (or fewer) patients
    num_patients = len(images_list)
    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20,12))
    axes = axes.ravel()
    for i in range(num_patients):
        img = images_list[i]
        pred = preds_list[i]
        pname = names_list[i]

        mid = img.shape[2]//2
        img_slice = img[:,:,mid]
        pred_slice = pred[:,:,mid]

        axes[i].imshow(img_slice, cmap='gray')
        overlay = np.zeros((*img_slice.shape,3))
        overlay[pred_slice==1] = [1,0,0]
        axes[i].imshow(overlay, alpha=0.3)
        axes[i].set_title(f'{pname}')
        axes[i].axis('off')

    for j in range(num_patients, rows*cols):
        axes[j].axis('off')

    plt.tight_layout()
    bulk_plot_path = os.path.join(OUTPUT_DIR, 'all_patients_overlay.png')
    plt.savefig(bulk_plot_path)
    print(f"All patients overlay plot saved at {bulk_plot_path}")

if __name__ == '__main__':
    main()
