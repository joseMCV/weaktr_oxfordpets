import torch
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from pathlib import Path
from PIL import Image
from torchvision.datasets import VisionDataset
import os
from utils.utils import PetSegmentationDataset, compute_iou

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Paths ---
    model_path = Path("Models")
    supervised_model_path = model_path / "segmentation_best_sup.pth"
    ws_model_path = model_path / "segmentation_best_ws.pth"
    image_dir = Path("Data/Split/test/images")
    gt_mask_dir = Path("Data/annotations/trimaps")

    # --- Transforms ---
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    # --- Dataset and Loader ---
    dataset = PetSegmentationDataset(image_dir, mask_dir=gt_mask_dir, gt_mask_dir=gt_mask_dir,
                                    transform=img_transform, target_transform=mask_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Load Models ---
    def load_model(path):
        model = lraspp_mobilenet_v3_large(weights=None, num_classes=1)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval().to(device)
        return model

    sup_model = load_model(supervised_model_path)
    ws_model = load_model(ws_model_path)

    # --- Evaluation ---
    intersection_sup = 0.0
    union_sup = 0.0
    intersection_ws = 0.0
    union_ws = 0.0
    samples = []

    def unnormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img_tensor * std + mean).clamp(0, 1)
    TP_sup, FP_sup, FN_sup, TN_sup = 0, 0, 0, 0
    TP_ws, FP_ws, FN_ws, TN_ws = 0, 0, 0, 0

    with torch.no_grad():
        for idx, (img, _, gt_mask) in enumerate(loader):
            img = img.to(device)
            gt_mask = gt_mask.to(device).bool()

            pred_sup = torch.sigmoid(sup_model(img)['out']) > 0.5
            pred_ws = torch.sigmoid(ws_model(img)['out']) > 0.5

            # --- Supervised ---
            TP_sup += ((pred_sup == 1) & (gt_mask == 1)).sum().item()
            FP_sup += ((pred_sup == 1) & (gt_mask == 0)).sum().item()
            FN_sup += ((pred_sup == 0) & (gt_mask == 1)).sum().item()
            TN_sup += ((pred_sup == 0) & (gt_mask == 0)).sum().item()

            # --- Weakly-Supervised ---
            TP_ws += ((pred_ws == 1) & (gt_mask == 1)).sum().item()
            FP_ws += ((pred_ws == 1) & (gt_mask == 0)).sum().item()
            FN_ws += ((pred_ws == 0) & (gt_mask == 1)).sum().item()
            TN_ws += ((pred_ws == 0) & (gt_mask == 0)).sum().item()

            # Save some samples
            if len(samples) < 5 and random.random() < 0.3:
                samples.append((img.cpu(), gt_mask.cpu(), pred_sup.cpu(), pred_ws.cpu()))

    # --- Metric Computation ---
    def compute_metrics(TP, FP, FN, TN):
        iou = TP / (TP + FP + FN + 1e-6)
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        return iou, dice, accuracy, precision, recall

    metrics_sup = compute_metrics(TP_sup, FP_sup, FN_sup, TN_sup)
    metrics_ws = compute_metrics(TP_ws, FP_ws, FN_ws, TN_ws)

    # --- Print Results ---
    def print_metrics(name, metrics):
        iou, dice, acc, prec, recall = metrics
        print(f"\n{name} Model:")
        print(f"  IoU           : {iou:.4f}")
        print(f"  Dice Coeff.   : {dice:.4f}")
        print(f"  Accuracy      : {acc:.4f}")
        print(f"  Precision     : {prec:.4f}")
        print(f"  Recall        : {recall:.4f}")
        print("="*50)

    print("Segmentation Metrics on Test Set:")
    print_metrics("Supervised", metrics_sup)
    print_metrics("Weakly-Supervised", metrics_ws)

    # --- Visualize Samples ---
    for img, gt, pred_sup, pred_ws in samples:
        img_np = unnormalize(img.squeeze()).permute(1, 2, 0).numpy()
        gt_np = gt.squeeze().numpy()
        sup_np = pred_sup.squeeze().numpy()
        ws_np = pred_ws.squeeze().numpy()

        iou_sup = ((sup_np * gt_np).sum() + 1e-6) / (((sup_np + gt_np) >= 1).sum() + 1e-6)
        iou_ws = ((ws_np * gt_np).sum() + 1e-6) / (((ws_np + gt_np) >= 1).sum() + 1e-6)

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Original Image")
        axs[1].imshow(gt_np, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(sup_np, cmap='gray')
        axs[2].set_title(f"Supervised Prediction\nIoU: {iou_sup:.3f}")
        axs[3].imshow(ws_np, cmap='gray')
        axs[3].set_title(f"Weakly-Supervised Prediction\nIoU: {iou_ws:.3f}")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
