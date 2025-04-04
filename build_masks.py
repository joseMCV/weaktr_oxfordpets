import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import timm
from pathlib import Path
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import DecoderHead, generate_mask

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    model_path = Path("Models")
    image_dir = Path("Data/Split/train/images")
    cam_dir = Path("FineCAMs")
    vit_path = model_path / "vit_pet_classifier_best.pth"
    decoder_path = model_path / "decoder_best.pth"
    output_dir = Path("RefinedMasks")  # NEW: clear name for output
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Threshold Config ---
    loss_threshold = 0.28
    use_finecam_only = False
    plot = True

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # --- Load Models ---
    vit_model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=37)
    vit_model.load_state_dict(torch.load(vit_path, map_location=device))
    vit_model.eval().to(device)

    decoder = DecoderHead(input_dim=192).to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()

    # --- Optional Visualization ---
    if plot:
        sample_files = sorted([f for f in image_dir.iterdir() if f.suffix == ".jpg"])[:5]
        for f in sample_files:
            cam_f = cam_dir / f.with_suffix(".npy").name
            if not cam_f.exists():
                continue
            mask, img, fine_cam, loss_map = generate_mask(
                f, cam_f, loss_threshold, transform, vit_model, decoder, use_finecam_only)
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(img)
            axs[0].set_title("Original Image")
            axs[1].imshow(fine_cam, cmap="jet")
            axs[1].set_title("Fine CAM")
            axs[2].imshow(loss_map, cmap="hot")
            axs[2].set_title("Loss Map")
            axs[3].imshow(mask, cmap="gray")
            axs[3].set_title("Refined Mask")
            for ax in axs:
                ax.axis("off")
            plt.tight_layout()
            plt.show()

    # --- Save All Refined Masks ---
    for f in tqdm(sorted([f for f in image_dir.iterdir() if f.suffix == ".jpg"])):
        cam_f = cam_dir / f.with_suffix(".npy").name
        if not cam_f.exists():
            continue
        mask, _, _, _ = generate_mask(
            f, cam_f, loss_threshold, transform, vit_model, decoder, use_finecam_only)
        Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / f.with_suffix(".png").name)
