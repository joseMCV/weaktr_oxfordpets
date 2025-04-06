import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from utils.utils import DecoderTrainingDataset, DecoderHead, compute_iou, set_seed

def main(epochs=15, seed=42):
    print("Starting decoder training...")

    # Set seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    model_path = Path("Models")
    image_dir = Path("Data/Split/train/images")
    cam_dir = Path("FineCAMs")
    cam_dir.mkdir(exist_ok=True)
    
    # Hyperparams
    batch_size = 64
    lr = 1e-2

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # Load pretrained ViT
    print("Loading frozen ViT encoder...")
    vit_model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=37)
    vit_model.load_state_dict(torch.load(model_path / "vit_pet_classifier_best.pth", map_location=device, weights_only=True))
    vit_model.eval().to(device)

    # Dataset
    dataset = DecoderTrainingDataset(image_dir, cam_dir, vit_model, transform=transform, device=device)
    print(f"Dataset loaded with {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Decoder
    decoder = DecoderHead(input_dim=192).to(device)
    decoder.apply(lambda m: nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # Training loop
    best_val_iou = 0.0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        decoder.train()
        total_loss = 0.0

        for batch_idx, (patch_tokens, fine_cams) in enumerate(train_loader):
            patch_tokens = patch_tokens.to(device)
            fine_cams = fine_cams.to(device)

            # Forward pass
            preds = decoder(patch_tokens)
            loss_per_pixel = loss_fn(preds, fine_cams)
            weights = fine_cams.clamp(0, 1)
            masked_loss = (loss_per_pixel * weights).sum() / weights.sum()

            # Backprop
            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()

            total_loss += masked_loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        decoder.eval()
        val_loss, val_iou = 0.0, 0.0

        with torch.no_grad():
            for patch_tokens, fine_cams in val_loader:
                patch_tokens = patch_tokens.to(device)
                fine_cams = fine_cams.to(device)

                preds = decoder(patch_tokens)
                loss_per_pixel = loss_fn(preds, fine_cams)
                weights = fine_cams.clamp(0, 1)
                masked_loss = (loss_per_pixel * weights).sum() / weights.sum()
                val_loss += masked_loss.item()
                val_iou += compute_iou(preds, fine_cams, threshold=0.5)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(decoder.state_dict(), model_path / "decoder_best.pth")
            print(f"Saved new best model (IoU: {best_val_iou:.4f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(decoder.state_dict(), model_path / "decoder_best_loss.pth")
            print(f"Saved model with best loss (Loss: {best_val_loss:.4f})")

    print("Training complete.")
