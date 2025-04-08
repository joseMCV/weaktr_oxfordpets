import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch import optim
from pathlib import Path

from utils.utils import OxfordPetBreedDataset, train_one_epoch, validate, set_seed
from torchvision import transforms

def main(epochs=15, seed=42, vit_model=None, transform=None):
    seed = seed
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path("Models")
    model_path.mkdir(exist_ok=True)
    data_path = Path("Data/Split/train/images")
    #  Load Dataset and Split
    dataset = OxfordPetBreedDataset(data_path, transform=transform)
    # Split Dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Load ViT Backbone
    num_classes = len(dataset.class_names)
    if vit_model == 'tiny':
        model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    elif vit_model == 'small':
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
    else:
        raise ValueError("num_patches must be either 16 or 32")
    model.head = nn.Linear(model.head.in_features, num_classes)  # Replace classification head

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    # Load ViT backbone
    num_classes = len(dataset.class_names)
    best_acc = 0.0
    epochs = epochs

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"Models/{vit_model}_vit_pet_classifier_best.pth")
            print(f" New best model saved with val_acc={val_acc:.4f}")