import torch
from pathlib import Path
import shutil
from utils.utils import set_seed

def main(seed=42):
    seed = seed
    set_seed(seed)
    # Input image directory
    image_dir = Path("Data/images")

    # Output directories
    train_dir = Path("Data/Split/train/images")
    test_dir = Path("Data/Split/test/images")

    # Create folders
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted([f for f in image_dir.glob("*.jpg")])
    total = len(image_files)
    test_ratio = 0.2
    test_size = int(total * test_ratio)

    # Generate random split indices using PyTorch
    indices = torch.randperm(total)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Split files based on indices
    train_files = [image_files[i] for i in train_indices]
    test_files = [image_files[i] for i in test_indices]

    # Copy files
    for file in train_files:
        shutil.copy(file, train_dir / file.name)

    for file in test_files:
        shutil.copy(file, test_dir / file.name)

    print(f"Copied {len(train_files)} training images and {len(test_files)} test images.")
