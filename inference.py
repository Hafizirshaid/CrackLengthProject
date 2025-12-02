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

from crack_seg import compute_iou, compute_dice, hybrid_loss
import argparse
from models.vit_based import CrackSegMixtureModel as ViTBasedSegModel
from pathlib import Path


def save_segmentation_visualization(
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    sample_name: str,
    iou: float,
    dice: float,
    save_dir: os.PathLike,
    dpi: int = 150,
) -> Path:
    """
    Saves a triptych visualization (image, GT mask, prediction) and returns the path.
    Expects gt_mask/pred_mask arrays to share the same spatial size as the PIL image.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"Input Image\n{sample_name}")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth\nMask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(f"Prediction Mask\nIoU: {iou*100:.1f}%  Dice: {dice*100:.1f}%")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = save_dir / f"{Path(sample_name).stem}_viz.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved segmentation visualization to {out_path}")
    return out_path

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


# ============================================================
# INFERENCE ON TEST SET
# ============================================================
def evaluate_test_set(model_path, root_dir, img_size=256, args=None):
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
    if args.model == 'unet':
        model = UNet().to(device)
    elif args.model == 'vit_based':
        #from vit_based import ViTBasedModel  # Assuming the ViT-based model is defined in vit_based.py
        model = ViTBasedSegModel().to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
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
    plot_dir = os.path.join(root_dir, "inference_plots", args.model)
    os.makedirs(plot_dir, exist_ok=True)
    
    for fname in tqdm(test_imgs):
        img_path = os.path.join(test_img_dir, fname)
        ann_path = os.path.join(test_ann_dir, fname)
        count += 1
        if count == 10:
            break
            
        # Load original image + mask for visualization
        original_img = Image.open(img_path).convert("RGB")
        # original_mask = Image.open(ann_path).convert("L")
        
        # # Load and preprocess image + mask for inference
        img = img_tf(original_img).unsqueeze(0).to(device)
        # --------- Load mask (correct) ---------
        original_mask = Image.open(ann_path).convert("L")

        mask_np = np.array(original_mask)

        # cracks are dark (near 0), background is light (near 255)
        # treat dark pixels as 1 (crack), light pixels as 0
        mask_np = (mask_np < 128).astype(np.uint8)    # <-- key change

        mask = Image.fromarray(mask_np)
        mask = mask.resize((img_size, img_size), Image.NEAREST)

        mask = torch.from_numpy(np.array(mask)).float()   # (H,W)
        mask = mask.unsqueeze(0)   


        #print('mask sum:', mask.sum().item())
        #print('unique mask values:', torch.unique(mask))

        with torch.no_grad():
            logits = model(img)
            
            # Get prediction
            pred_prob = torch.sigmoid(logits)
            pred_mask = (pred_prob > 0.5).float()

            # print(f"sum of pred_mask: {pred_mask.sum().item()}")
            # print(f"unque pred_mask values: {torch.unique(pred_mask)}")

            # loss for reference
            total_loss += hybrid_loss(logits, mask).item()

            # IoU
            iou = compute_iou(logits, mask)
            total_iou += iou

            # Dice
            dice = compute_dice(logits, mask)
            total_dice += dice
            
            pred_np = pred_mask.squeeze().cpu().numpy()
            original_mask_size = original_mask.size  # (width, height)
            pred_resized = (
                np.array(
                    Image.fromarray((pred_np * 255).astype(np.uint8)).resize(
                        original_mask_size, Image.NEAREST
                    )
                )
                / 255.0
            )

            save_segmentation_visualization(
                original_img,
                np.array(mask_np),
                pred_resized,
                fname,
                iou,
                dice,
                plot_dir,
                dpi=150,
            )

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

    parser = argparse.ArgumentParser(description="U-Net Inference on Test Set")
    parser.add_argument('--model_path', type=str, default='checkpoints/unet_best.pth', help='Path to the trained U-Net model')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'vit_based'], help='Model type to use')
    args = parser.parse_args()
    
    # ------ TEST ------
    evaluate_test_set(
        model_path=args.model_path,
        root_dir="dataset/omnicrack30k",
        img_size=256,
        args=args
    )
