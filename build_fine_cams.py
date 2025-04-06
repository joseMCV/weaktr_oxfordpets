import os

import torch
import timm
import numpy as np

from torchvision import transforms
from pathlib import Path
from PIL import Image

from tqdm import tqdm
from types import MethodType

from utils.utils import get_patch_tokens, extract_cls2patch_attention, adaptive_attention_fusion, generate_coarse_cam, generate_fine_cam

def main():
    
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
    model_path = model_path.joinpath("vit_pet_classifier_best.pth")
    finecam_dir = Path("FineCAMs")
    finecam_dir.mkdir(exist_ok=True)
    num_classes = 37
    img_size = 224
    patch_size = 16

    # ---- Load model ----
    model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # ---- Monkey-patch attention to capture maps ----
    attention_maps = []
    def store_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attention_maps.append(attn.detach())
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    for blk in model.blocks:
        blk.attn.forward = MethodType(store_attn, blk.attn)
    
    # ---- Process and Save Fine CAMs ----
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    for filename in tqdm(image_files):
        attention_maps.clear()
        img_path = image_dir / filename
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model(input_tensor)
            patch_tokens = get_patch_tokens(model, input_tensor)
            cls2patch_all = extract_cls2patch_attention(attention_maps)
            aaf_maps = adaptive_attention_fusion(cls2patch_all, patch_tokens, num_classes)
            class_weights = model.head.weight.data.t()
            coarse_cams = generate_coarse_cam(patch_tokens, class_weights)
            pred_class = model(input_tensor).argmax(dim=1).item()
            fine_cam = generate_fine_cam(coarse_cams, aaf_maps, pred_class)

        save_path = finecam_dir / f"{filename.replace('.jpg', '.npy')}"
        np.save(save_path, fine_cam)
