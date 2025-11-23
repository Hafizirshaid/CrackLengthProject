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
# U-NET MODEL
# ============================================================
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


def compute_iou(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = target.unsqueeze(1)

    inter = (pred * target).sum((2,3))
    union = ((pred + target) > 0).float().sum((2,3))

    return (inter / (union + 1e-6)).mean().item()


def compute_dice(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = target.unsqueeze(1)

    inter = (pred * target).sum((2,3))
    union = pred.sum((2,3)) + target.sum((2,3))

    dice = (2 * inter) / (union + 1e-6)
    return dice.mean().item()


# ============================================================
# INFERENCE ON TEST SET
# ============================================================
def evaluate_test_set(model_path, root_dir, img_size=256):
    """
    Evaluates U-Net on the test dataset using IoU and Dice.
    Assumes:
        root_dir/images/test/*.png
        root_dir/annotations/test/*.png
    """
    device = (
         "cpu"
    )

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Same transforms as training
    img_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    mask_tf = T.Resize((img_size, img_size),
                       interpolation=T.InterpolationMode.NEAREST)

    # Directories
    test_img_dir = os.path.join(root_dir, "images", "test")
    test_ann_dir = os.path.join(root_dir, "annotations", "test")

    test_imgs = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])

    total_iou = 0
    total_dice = 0
    total_loss = 0

    print(f"Evaluating {len(test_imgs)} test images...")
    count = 0
    
    # Create output directory for plots
    plot_dir = os.path.join(root_dir, "inference_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    for fname in tqdm(test_imgs):
        img_path = os.path.join(test_img_dir, fname)
        ann_path = os.path.join(test_ann_dir, fname)
        count += 1
        if count == 10:
            break
            
        # Load original image + mask for visualization
        original_img = Image.open(img_path).convert("RGB")
        original_mask = Image.open(ann_path).convert("L")
        
        # Load and preprocess image + mask for inference
        img = img_tf(original_img).unsqueeze(0).to(device)
        mask = mask_tf(original_mask)
        mask = (np.array(mask) > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W)

        with torch.no_grad():
            logits = model(img)
            
            # Get prediction
            pred_prob = torch.sigmoid(logits)
            pred_mask = (pred_prob > 0.5).float()

            # loss for reference
            total_loss += hybrid_loss(logits, mask).item()

            # IoU
            iou = compute_iou(logits, mask)
            total_iou += iou

            # Dice
            dice = compute_dice(logits, mask)
            total_dice += dice
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_img)
            axes[0].set_title(f'Input Image\n{fname}')
            axes[0].axis('off')
            
            # Ground truth mask
            axes[1].imshow(np.array(original_mask), cmap='gray')
            axes[1].set_title(f'Ground Truth\nMask')
            axes[1].axis('off')
            
            # Prediction
            pred_np = pred_mask.squeeze().cpu().numpy()
            # resize pred_np to match original_mask size
            original_mask_size = original_mask.size  # (width, height)
            pred_resized = np.array(Image.fromarray((pred_np * 255).astype(np.uint8)).resize(original_mask_size, Image.NEAREST)) / 255.0

            axes[2].imshow(pred_resized, cmap='gray')
            axes[2].set_title(f'Prediction Mask')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plot_dir, f"inference_{fname.replace('.png', '.png')}")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory

    # N = len(test_imgs)
    N = 10
    print("\n======= TEST RESULTS =======")
    print(f"Test Loss: {total_loss / N:.4f}")
    print(f"Test IoU : {total_iou / N:.4f}")
    print(f"Test Dice: {total_dice / N:.4f}")
    print(f"Plots saved to: {plot_dir}")
    print("============================\n")

    return total_loss / N, total_iou / N, total_dice / N



if __name__ == "__main__":

    # ------ TEST ------
    evaluate_test_set(
        model_path="unet_best.pth",
        root_dir="/Users/hafezirshaid/Desktop/CrackDetectionProject/omnicrack30k",
        img_size=256
    )