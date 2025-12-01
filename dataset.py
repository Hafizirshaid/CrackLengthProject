import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ============================================================
# DATASET
# ============================================================
class OmniCrackDataset(Dataset):
    def __init__(self, root_dir, split="training", img_size=256):
        """
        Folder structure:
            root_dir/
                images/training/*.png
                annotations/training/*.png
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        img_dir = os.path.join(root_dir, "images", split)
        ann_dir = os.path.join(root_dir, "annotations", split)

        # Read all images
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        self.ann_dir = ann_dir

        # Transforms
        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],
                        [0.229,0.224,0.225])
        ])
        
        self.mask_tf = T.Resize((img_size, img_size),
                                interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        fname = os.path.basename(img_path)

        ann_path = os.path.join(self.ann_dir, fname)
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Missing annotation: {ann_path}")

        # Load image + mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L")

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask < 128).astype(np.float32)   # cracks = 1, background = 0

        mask = torch.from_numpy(mask)         # (H, W)

        
        return img, mask
