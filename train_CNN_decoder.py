import torch
import torch.nn as nn
import torch.optim as optim
import timm


from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from utils.utils import DecoderTrainingDataset, DecoderHead, compute_iou

def main(epochs=15, seed=42):
    seed = seed
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    model_path = Path("Models")
    image_dir= Path("Data/Split/train/images")
    finecam_dir = Path("FineCAMs")
    finecam_dir.mkdir(exist_ok=True)
    cam_dir = Path("FineCAMs")
    batch_size = 64
    epochs = 10
    lr = 1e-4

    # ---- Image Transform ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # ---- Load Frozen ViT Encoder ----
    vit_model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=37)
    vit_model.load_state_dict(torch.load(model_path.joinpath("vit_pet_classifier_best.pth"), map_location=device, weights_only=True))
    vit_model.eval().to(device)

    # ---- Dataset & DataLoader ----
    dataset = DecoderTrainingDataset(image_dir, cam_dir, vit_model, transform=transform, device=device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Initialize Decoder ----
    decoder = DecoderHead(input_dim=192).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # we'll mask loss manually

    # ---- Training Loop ----
    best_val_iou = 0.0
    for epoch in range(epochs):
        decoder.train()
        total_loss = 0.0

        for patch_tokens, fine_cams in train_loader:
            patch_tokens = patch_tokens.to(device)
            fine_cams = fine_cams.to(device)

            preds = decoder(patch_tokens)
            # mask = (fine_cams > 0.1).float()
            loss_per_pixel = loss_fn(preds, fine_cams)
            # masked_loss = (loss_per_pixel * mask).sum() / (mask.sum() + 1e-6)
            
            weights = fine_cams.clamp(0, 1)  # 0–1 confidence
            masked_loss = (loss_per_pixel * weights).sum() / weights.sum()

            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()
            total_loss += masked_loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- Validation ---
        decoder.eval()
        val_iou = 0.0
        with torch.no_grad():
            for patch_tokens, fine_cams in val_loader:
                patch_tokens = patch_tokens.to(device)
                fine_cams = fine_cams.to(device)
                preds = decoder(patch_tokens)
                val_iou += compute_iou(preds, fine_cams)

        val_iou /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(decoder.state_dict(), "Models/decoder_best.pth")
            print(f"Saved new best decoder (Val IoU: {val_iou:.4f})")