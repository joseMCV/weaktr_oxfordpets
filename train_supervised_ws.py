import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import PetSegmentationDataset, compute_iou
import torchvision.models.segmentation as models

def main(seed=42):
    seed = seed
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path("Models")
    image_dir= Path("Data/Split/train/images")
    finecam_dir = Path("FineCAMs")
    finecam_dir.mkdir(exist_ok=True)
    cam_dir = Path("FineCAMs")

    decoder_path = model_path.joinpath("decoder_best.pth")
    vit_path = model_path.joinpath("vit_pet_classifier_best.pth")
    
    gt_mask_dir = Path("Data/annotations/trimaps")
    mask_dir = Path("RefinedMasks")
    model_path = Path("Models")

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

    epochs = 10

    # --- Dataset & Loader ---
    dataset = PetSegmentationDataset(image_dir, mask_dir, gt_mask_dir,
                                    transform=img_transform,
                                    target_transform=mask_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2, drop_last=True)

    # --- Model: U-Net from torchvision's segmentation model (deeplabv3 with MobileNet as U-Net proxy) ---
    model = models.lraspp_mobilenet_v3_large(pretrained=False, num_classes=1)
    model = model.to(device)

    # --- Loss and Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_val_iou = 0.0
    best_val_loss = float('inf')
    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_iou_pseudo = 0.0
        train_iou_gt = 0.0

        for imgs, pseudo_masks, gt_masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs = imgs.to(device)
            pseudo_masks = pseudo_masks.to(device)
            gt_masks = gt_masks.to(device)

            preds = model(imgs)['out']
            loss = criterion(preds, pseudo_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou_pseudo += compute_iou(torch.sigmoid(preds), pseudo_masks)
            train_iou_gt += compute_iou(torch.sigmoid(preds), gt_masks)  # NEW: IoU vs GT

        model.eval()
        val_loss = 0.0
        val_iou_gt = 0.0
        val_iou_pseudo = 0.0

        with torch.no_grad():
            for imgs, pseudo_masks, gt_masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs = imgs.to(device)
                pseudo_masks = pseudo_masks.to(device)
                gt_masks = gt_masks.to(device)

                preds = model(imgs)['out']
                loss = criterion(preds, pseudo_masks)

                val_loss += loss.item()
                val_iou_gt += compute_iou(torch.sigmoid(preds), gt_masks)
                val_iou_pseudo += compute_iou(torch.sigmoid(preds), pseudo_masks)  # NEW

        # Averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou_pseudo = train_iou_pseudo / len(train_loader)
        avg_train_iou_gt = train_iou_gt / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou_gt = val_iou_gt / len(val_loader)
        avg_val_iou_pseudo = val_iou_pseudo / len(val_loader)

        print(f"Epoch {epoch}:")
        print(f"  Train Loss     : {avg_train_loss:.4f}")
        print(f"  Train IoU (Pseudo): {avg_train_iou_pseudo:.4f} | Train IoU (GT): {avg_train_iou_gt:.4f}")
        print(f"  Val   Loss     : {avg_val_loss:.4f}")
        print(f"  Val   IoU (Pseudo): {avg_val_iou_pseudo:.4f}  | IoU (GT) : {avg_val_iou_gt:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path / "segmentation_best_ws.pth")
            print(f"Saved new best model (Val Loss: {avg_val_loss:.4f})")

