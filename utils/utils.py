import torch
import os
import random

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

from PIL import Image
from tqdm import tqdm
from pathlib import Path

seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)
#%%
# # CLASSIFICATION TASK
class OxfordPetBreedDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.class_names = sorted(list({ '_'.join(f.split('_')[:-1]) for f in self.image_files }))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        label_name = '_'.join(filename.split('_')[:-1])
        label = self.class_to_idx[label_name]
        if self.transform:
            img = self.transform(img)
        return img, label


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


# Validation Loop
def validate(model, dataloader, criterion):
    model.eval()
    total, correct = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    return val_loss / len(dataloader), acc
#%% Build Fine CAMs
# ---- Helper Functions ----
def extract_cls2patch_attention(attn_maps):
    cls2patch_all = []
    for attn in attn_maps:
        cls2patch = attn[0, :, 0, 1:]  # shape: (H, N)
        cls2patch_all.append(cls2patch)
    return cls2patch_all

def get_patch_tokens(model, x):
    x = model.patch_embed(x)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    for blk in model.blocks:
        x = blk(x)
    return x[:, 1:]

def adaptive_attention_fusion(cls2patch_all, patch_tokens, num_classes):
    fusion_maps = []
    token_feats = patch_tokens.squeeze(0)
    norm_feats = F.normalize(token_feats, dim=1)
    for cls in range(num_classes):
        weights = []
        for layer_attn in cls2patch_all:
            layer_attn = layer_attn.to(norm_feats.device)
            sim = layer_attn @ norm_feats
            sim = sim.mean(dim=1)
            weights.append(F.softmax(sim, dim=0))
        weights = torch.stack(weights)
        fused_attn = sum(w.unsqueeze(1) * a.to(w.device) for w, a in zip(weights, cls2patch_all))
        fusion_maps.append(fused_attn)
    return fusion_maps

def generate_coarse_cam(patch_tokens, class_weights, img_size=224, patch_size=16):
    cams = torch.matmul(patch_tokens, class_weights)
    cams = cams.squeeze(0).permute(1, 0)
    h = w = img_size // patch_size
    cams = cams.reshape(-1, h, w)
    cams = F.interpolate(cams.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False)
    return cams.squeeze(0)

def generate_fine_cam(coarse_cams, aaf_maps, class_idx, img_size=224, patch_size=16):
    h = w = img_size // patch_size
    coarse = coarse_cams[class_idx].unsqueeze(0).unsqueeze(0)
    if coarse.shape[-1] != h:
        coarse = F.interpolate(coarse, size=(h, w), mode='bilinear', align_corners=False)
    attn = aaf_maps[class_idx]
    attn = attn.mean(dim=0)
    attn_map = attn.reshape(h, w)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-6)
    coarse = (coarse - coarse.min()) / (coarse.max() + 1e-6)
    fine_cam = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0).to(coarse.device) * coarse
    fine_cam = F.interpolate(fine_cam, size=(img_size, img_size), mode='bilinear', align_corners=False)
    return fine_cam.squeeze().cpu().numpy()
#%% DECODER
class DecoderTrainingDataset(Dataset):
    def __init__(self, image_dir, cam_dir, vit_model, transform=None, device = 'cuda'):
        self.image_dir = Path(image_dir)
        self.device = device
        self.cam_dir = Path(cam_dir)
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
        self.vit = vit_model
        self.vit.eval()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = self.image_dir / filename
        cam_path = self.cam_dir / filename.replace(".jpg", ".npy")

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        else:
            raise ValueError("Transform must be defined")

        # Get patch tokens (detach to prevent gradient to ViT)
        with torch.no_grad():
            tokens = self.vit.patch_embed(img_tensor)
            cls_token = self.vit.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_token, tokens), dim=1)
            tokens = self.vit.pos_drop(tokens + self.vit.pos_embed)
            for blk in self.vit.blocks:
                tokens = blk(tokens)
            patch_tokens = tokens[:, 1:]  # remove cls token
            patch_tokens = patch_tokens.reshape(1, 14, 14, -1).permute(0, 3, 1, 2).squeeze(0)  # [C, 14, 14]

        # Load fine CAM
        fine_cam = np.load(cam_path)
        fine_cam = torch.tensor(fine_cam, dtype=torch.float32).unsqueeze(0)  # [1, 224, 224]

        return patch_tokens, fine_cam

class SmallDecoderHead(nn.Module):
    def __init__(self, input_dim=192):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 14 → 28
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 28 → 56
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 56 → 224
            nn.Conv2d(16, 1, kernel_size=1)  # Output logits
        )

    def forward(self, x):
        return self.decoder(x)

class DecoderHead(nn.Module):
    def __init__(self, input_dim=192):  # 192 = ViT tiny patch token dim
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 14 -> 28
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 28 -> 56
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 56 -> 224
            nn.Conv2d(32, 1, kernel_size=1)  # Output logits
        )

    def forward(self, x):
        return self.decoder(x)

class LargeDecoderHead(nn.Module):
    def __init__(self, input_dim=192):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 14 → 28
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 28 → 56
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 56 → 224
            nn.Conv2d(64, 1, kernel_size=1)  # Output logits
        )

    def forward(self, x):
        return self.decoder(x)


def compute_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) >= 1).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

#%% Building Masks
# --- Helper function ---
def get_patch_tokens_f(img_tensor, vit_model):
    with torch.no_grad():
        tokens = vit_model.patch_embed(img_tensor)
        cls_token = vit_model.cls_token.expand(img_tensor.size(0), -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = vit_model.pos_drop(tokens + vit_model.pos_embed)
        for blk in vit_model.blocks:
            tokens = blk(tokens)
        patch_tokens = tokens[:, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        return patch_tokens

def generate_mask(image_path, cam_path, threshold, transform, vit_model, decoder, use_finecam_only):
    img = Image.open(image_path).convert("RGB")
    fine_cam = np.load(cam_path)

    if use_finecam_only:
        mask = (fine_cam <= threshold).astype(np.uint8)
        return mask, img, fine_cam, fine_cam

    img_tensor = transform(img).unsqueeze(0).to(device)
    fine_cam_tensor = torch.tensor(fine_cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    patch_tokens = get_patch_tokens_f(img_tensor, vit_model)
    preds = decoder(patch_tokens)

    loss_map = F.binary_cross_entropy_with_logits(preds, fine_cam_tensor, reduction='none')
    loss_map = loss_map.squeeze().detach().cpu().numpy()
    loss_map = (loss_map - loss_map.min()) / (loss_map.max() + 1e-6)

    mask = (loss_map <= threshold).astype(np.uint8)
    return mask, img, fine_cam, loss_map

#%% SUPERVISED MODEL TRAINING
# --- Custom Dataset ---
class PetSegmentationDataset(VisionDataset):
    def __init__(self, image_dir, mask_dir, gt_mask_dir, transform=None, target_transform=None):
        super().__init__(image_dir)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.gt_mask_dir = Path(gt_mask_dir)
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        gt_mask_name = img_name
        img = Image.open(self.image_dir / img_name).convert("RGB")
        mask = Image.open(self.mask_dir / img_name.replace(".jpg", ".png"))
        gt_mask = Image.open(self.gt_mask_dir / gt_mask_name.replace(".jpg", ".png"))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
            gt_mask = self.target_transform(gt_mask)

        # mask = (mask * 255).byte()
        # Pseudo mask fix
        mask = (mask == 1.0).float()

        # Ground truth (pet + boundary)
        gt_mask = gt_mask.squeeze(0) * 255
        gt_mask = ((gt_mask == 2) | (gt_mask == 3)).float().unsqueeze(0)

        return img, mask, gt_mask
