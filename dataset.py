import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List
import csv
import torchvision.transforms as T
import matplotlib.pyplot as plt



class OmniCrackDataset(Dataset):
    def __init__(self, root_dir, split="training", img_size=256, transform=None):

        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        img_dir = os.path.join(root_dir, "images", split)
        ann_dir = os.path.join(root_dir, "annotations", split)
        centerline_dir = os.path.join(root_dir, "centerlines", split)

        # Read all images
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        self.ann_dir = ann_dir
        self.centerline_dir = centerline_dir
        self.transform = transform
       
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        fname = os.path.basename(img_path)
        centerline_path = os.path.join(self.centerline_dir, fname)

        ann_path = os.path.join(self.ann_dir, fname)
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Missing annotation: {ann_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L")
        centerline = Image.open(centerline_path).convert("L")

        img, mask, centerline = self.transform(img, mask, centerline)

        mask = np.array(mask, dtype=np.uint8)
        mask = (mask < 128).astype(np.float32)   # cracks = 1, background = 0

        mask = torch.from_numpy(mask)         # (H, W)
        centerline = torch.from_numpy((np.array(centerline) < 128).astype("float32")).unsqueeze(0)
        
        return img, mask, centerline


class KaggleCrackLenDataset(Dataset):
    """
    Dataset for the Kaggle crack length data:
      crack_final/
        1-Segmentation/
          Original Image/
          Ground Truth/
        4-Crack Length/
          Crack Length.txt

    Returns:
      img:  (3, H, W) float32 in [0,1]
      mask: (1, H, W) float32 in {0,1}
      length: scalar float32 (crack length, cm)
    """

    def __init__(
        self,
        root: str,
        img_size: int = 256,
        split: str = "all",
        indices: Optional[List[int]] = None,
        img_ext: str = ".jpg",
        mask_ext: str = ".png",
    ):
        """
        Args:
            root: path to 'crack_final' folder.
            img_size: resize square size for both image and mask.
            split: for now "all". You can handle "train"/"val"/"test"
                   using 'indices' (pre-split indexes).
            indices: optional subset of rows (0-based).
            img_ext: extension for original images (".png" / ".jpg").
            mask_ext: extension for GT masks.
        """
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.split = split
        self.img_ext = img_ext
        self.mask_ext = mask_ext

        self.img_dir = os.path.join(root, "1-Segmentation", "Original Image")
        self.mask_dir = os.path.join(root, "1-Segmentation", "Ground Truth")

        lengths_path = os.path.join(root, "4-Crack Length", "Crack Length.txt")

        # ---- read Crack Length.txt ----
        # file looks like: "filename<TAB>Crack Length (cm)"
        self.records = []  # list of dicts: {"id": int, "length": float}
        with open(lengths_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # keys: "filename", "Crack Length (cm)"  :contentReference[oaicite:2]{index=2}
                fid = int(row["filename"])
                length_cm = float(row["Crack Length (cm)"])
                self.records.append({"id": fid, "length": length_cm})

        # optional subset of indices (0-based)
        if indices is not None:
            self.records = [self.records[i] for i in indices]

        # transforms
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # (3,H,W) in [0,1]
        ])
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),  # (1,H,W) in [0,1], but still 0/1 if input 0/255
        ])

    def __len__(self):
        return len(self.records)

    def _resolve_path(self, base_dir: str, fid: int, ext: str):
        """
        Try ext; if missing, try .png/.jpg fallback.
        """
        fname = str(fid) + ext
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            return path

        # fallback: try .png/.jpg if not found
        alt_exts = [".png", ".jpg", ".jpeg"]
        for e in alt_exts:
            alt = os.path.join(base_dir, str(fid) + e)
            if os.path.exists(alt):
                return alt
        raise FileNotFoundError(f"Could not find file for id={fid} in {base_dir}")

    def __getitem__(self, idx):
        rec = self.records[idx]
        fid = rec["id"]
        length_cm = rec["length"]

        # 1) load image & mask
        img_path = self._resolve_path(self.img_dir, fid, self.img_ext)
        mask_path = self._resolve_path(self.mask_dir, fid, self.mask_ext)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        img = self.img_transform(img)  # (3,H,W)
        mask = self.mask_transform(mask)  # (1,H,W) in [0,1] approx

        # ensure binary mask in {0,1} (if it's 0/255 originally)
        mask = (mask > 0.5).float()

        length = torch.tensor(length_cm, dtype=torch.float32)

        return img, mask, length