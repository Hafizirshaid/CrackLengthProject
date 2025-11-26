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

from models.vit_based import CrackSegMixtureModel as ViTBasedSegModel

import argparse

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


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)  # logits


# ============================================================
# LOSSES & METRICS
# ============================================================
def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    target = target.unsqueeze(1)

    inter = (probs * target).sum((2,3))
    union = probs.sum((2,3)) + target.sum((2,3))

    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()


def hybrid_loss(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask.unsqueeze(1))
    d = dice_loss(logits, mask)
    return bce + d


# def compute_iou(logits, target):
#     pred = (torch.sigmoid(logits) > 0.5).float()
#     target = target.unsqueeze(1)

#     inter = (pred * target).sum((2,3))
#     union = ((pred + target) > 0).float().sum((2,3))

#     return (inter / (union + 1e-6)).mean().item()

def compute_iou(logits, target):
    """
    logits: (B,1,H,W)
    target: (B,H,W) with values 0 or 1
    """
    # binarize predictions
    pred = (torch.sigmoid(logits) > 0.5).float()   # (B,1,H,W)

    # make target (B,1,H,W)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    # foreground intersection & union (NO true negatives)
    inter = (pred * target).sum(dim=(2, 3))                # TP
    union = ((pred + target) > 0).float().sum(dim=(2, 3))  # TP+FP+FN

    # print('pred areas :', pred.sum(dim=(2,3)))
    # print('target areas :', target.sum(dim=(2,3)))
    # print('inter :', inter)
    # print('union :', union)
    iou = inter / (union + 1e-6)
    return iou.mean().item()



def compute_dice(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = target.unsqueeze(1)

    inter = (pred * target).sum((2,3))
    union = pred.sum((2,3)) + target.sum((2,3))

    dice = (2 * inter) / (union + 1e-6)
    return dice.mean().item()


# ============================================================
# TRAINING LOOP
# ============================================================
def train_unet(args=None):
    root = "dataset/omnicrack30k"   # TODO: CHANGE THIS
    img_size = 256
    batch_size = 4
    lr = 1e-4
    epochs = args.epochs

    device = "cuda" if torch.cuda.is_available() else "mps"
    print("Device:", device)

    train_ds = OmniCrackDataset(root, split="training", img_size=img_size)
    val_ds   = OmniCrackDataset(root, split="validation", img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.model == 'unet':
        model = UNet().to(device)
    elif args.model == 'vit_based':
        model = ViTBasedSegModel().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_loss = float("inf")

    for epoch in range(1, epochs+1):
        print(f"\nEPOCH {epoch}/{epochs}")
        step = 0

        # ---------------- Train ----------------
        model.train()
        total = 0
        for imgs, masks in tqdm(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            opt.zero_grad()
            logits = model(imgs)
            loss = hybrid_loss(logits, masks)
            loss.backward()
            opt.step()

            total += loss.item()

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break  # stop early for this epoch
        train_loss = total / len(train_loader)

        # ---------------- Val ----------------
        model.eval()
        total = 0
        val_iou = 0
        val_dice = 0

        with torch.no_grad():
            step_val = 0
            for imgs, masks in tqdm(val_loader):
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                total += hybrid_loss(logits, masks).item()
                val_iou += compute_iou(logits, masks)
                val_dice += compute_dice(logits, masks)
                step_val += 1
                if args.max_steps is not None and step_val >= args.max_steps:
                    break  # stop early for this epoch
        val_loss = total / len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}  |  IoU: {val_iou:.4f}  |  Dice: {val_dice:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"checkpoints/{args.model}_best.pth")
            print(" -> Saved new BEST model!")

    # Save final model
    torch.save(model.state_dict(), f"checkpoints/{args.model}_last.pth")
    print("Training finished.")

    # ---------------- Plot Loss Curve ----------------
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("U-Net Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_curve_{args.model}.png", dpi=200)
    plt.show()

    print(f"Saved loss_curve_{args.model}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for crack segmentation")
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum training steps (overrides epochs if set)')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'vit_based'], help='Model type to train')

    args = parser.parse_args()

    train_unet(args)




