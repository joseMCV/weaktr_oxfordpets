import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.utils import PetSegmentationDataset, compute_iou

def main(seed=42):
    seed = seed
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    model_path = Path("Models")
    image_dir = Path("Data/Split/train/images")
    gt_mask_dir = Path("Data/annotations/trimaps")
    mask_dir = gt_mask_dir  # Dummy path, not used

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

    # --- Dataset & DataLoaders ---
    dataset = PetSegmentationDataset(image_dir, mask_dir, gt_mask_dir,
                                    transform=img_transform,
                                    target_transform=mask_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2, drop_last=True)

    # --- Model ---
    model = lraspp_mobilenet_v3_large(pretrained=False, num_classes=1)
    model = model.to(device)

    # --- Loss & Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    best_val_loss = float('inf')

    # --- Training Loop ---
    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_iou = 0.0, 0.0
        for imgs, _, gt_masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, gt_masks = imgs.to(device), gt_masks.to(device)
            preds = model(imgs)['out']
            loss = criterion(preds, gt_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += compute_iou(torch.sigmoid(preds), gt_masks)

        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, _, gt_masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs, gt_masks = imgs.to(device), gt_masks.to(device)
                preds = model(imgs)['out']
                loss = criterion(preds, gt_masks)

                val_loss += loss.item()
                val_iou += compute_iou(torch.sigmoid(preds), gt_masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Train IOU (GT) {avg_train_iou:.4f}")
        print(f"             Val Loss {avg_val_loss:.4f},   Val IOU (GT) {avg_val_iou:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path / "segmentation_best_sup.pth")
            print(f"Saved new best model (Val Loss: {avg_val_loss:.4f})")
