import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet
from models.vit_based import CrackSegMixtureModel as ViTBasedSegModel
from models.unet_resnet import get_unet_resnet50
from dataset import OmniCrackDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np
import random
from transform.crack_seg_transform import CrackSegTrainTransform, CrackSegValTransform

def set_seed(seed):
    """Sets the random seed for reproducibility across multiple libraries."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    target = target.unsqueeze(1)

    inter = (probs * target).sum((2, 3))
    union = probs.sum((2, 3)) + target.sum((2, 3))

    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def hybrid_loss(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask.unsqueeze(1))
    d = dice_loss(logits, mask)
    return bce + d

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


def train_unet(args=None):
    root = "/mnt/home/irshaid2/crack_seg/omnicrack30k"
    img_size = 256
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    print("=======================================")
    print(f"Training {args.model} for Crack Segmentation")
    print(f"learning rate: {lr}, batch size: {batch_size}, img size: {img_size}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model_name = f"{args.model}_lr_{lr}_bs_{batch_size}"
    os.makedirs("checkpoints", exist_ok=True)
    print("=======================================")
    
    train_transform = CrackSegTrainTransform(img_size=img_size)
    val_transform = CrackSegValTransform(img_size=img_size)

    train_ds = OmniCrackDataset(root, split="training", img_size=img_size, transform=train_transform)
    val_ds   = OmniCrackDataset(root, split="validation", img_size=img_size, transform=val_transform)
    test_ds = OmniCrackDataset(root, split="test", img_size=img_size, transform=val_transform)


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.model == 'unet':
        model = UNet().to(device)
    elif args.model == 'vit_based':
        model = ViTBasedSegModel().to(device)
    elif args.model == "unet_resnet":
        model = get_unet_resnet50(num_classes=1, pretrained=True, freeze_encoder=False, bilinear=True).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

    train_losses = []
    val_losses = []
    best_loss = float("inf")

    for epoch in range(1, epochs+1):
        print(f"\nEPOCH {epoch}/{epochs}")
        step = 0

        # ---------------- Train ----------------
        model.train()
        total = 0
        for imgs, masks, centerlines in tqdm(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            centerlines = centerlines.to(device)

            opt.zero_grad()
            logits = model(imgs)
            loss = hybrid_loss(logits, masks)
            loss.backward()
            opt.step()

            total += loss.item()

            step += 1
            # if args.max_steps is not None and step >= args.max_steps:
            #     break  # stop early for this epoch
        train_loss = total / len(train_loader)

        # ---------------- Val ----------------
        model.eval()
        total = 0
        val_iou = 0
        val_dice = 0

        with torch.no_grad():
            step_val = 0
            for imgs, masks, centerlines in tqdm(val_loader):
                imgs = imgs.to(device)
                masks = masks.to(device)
                centerlines = centerlines.to(device)

                logits = model(imgs)
                total += hybrid_loss(logits, masks).item()
                val_iou += compute_iou(logits, masks)
                val_dice += compute_dice(logits, masks)
                step_val += 1
                # if args.max_steps is not None and step_val >= args.max_steps:
                #     break  # stop early for this epoch
        val_loss = total / len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}  |  IoU: {val_iou:.4f}  |  Dice: {val_dice:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"checkpoints/{model_name}_epoch_{epochs}_aug_2_best.pth")
            print(" -> Saved new BEST model!")

        # ---------------- Plot Loss Curve ----------------
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.yscale("log")
        plt.ylabel("Loss")
        plt.title("U-Net Training Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"loss_curve_{model_name}_aug_2_epoch_{epochs}.png", dpi=200)
        plt.show()

    # Save final model
    torch.save(model.state_dict(), f"checkpoints/{model_name}_aug_2_last.pth")
    print("Training finished.")
    print(f"Saved loss_curve_{model_name}_aug_epoch_{epochs}.png")

    # test
    model.eval()
    test_iou = 0
    test_dice = 0
    test_centerline_iou_1px = 0
    test_centerline_iou_2px = 0
    test_centerline_iou_3px = 0
    test_centerline_iou_4px = 0

    with torch.no_grad():
        for imgs, masks, centerlines in tqdm(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            centerlines = centerlines.to(device)

            logits = model(imgs)
            test_iou += compute_iou(logits, masks)
            test_centerline_iou_1px += centerline_iou(logits, centerlines, 1)
            test_centerline_iou_2px += centerline_iou(logits, centerlines, 2)
            test_centerline_iou_3px += centerline_iou(logits, centerlines, 3)
            test_centerline_iou_4px += centerline_iou(logits, centerlines, 4)

            test_dice += compute_dice(logits, masks)

    print("=======================================")
    print(f"Test IoU: {test_iou / len(test_loader):.4f}")
    print(f"Test Centerline IoU (1px): {test_centerline_iou_1px / len(test_loader):.4f}")
    print(f"Test Centerline IoU (2px): {test_centerline_iou_2px / len(test_loader):.4f}")
    print(f"Test Centerline IoU (3px): {test_centerline_iou_3px / len(test_loader):.4f}")
    print(f"Test Centerline IoU (4px): {test_centerline_iou_4px / len(test_loader):.4f}")
    print(f"Test Dice: {test_dice / len(test_loader):.4f}")
    print(f"learning rate: {lr}, batch size: {batch_size}, img size: {img_size}")
    print("=======================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for crack segmentation")
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    # parser.add_argument('--max_steps', type=int, default=None, help='Maximum training steps (overrides epochs if set)')
    parser.add_argument('--model', type=str, default='unet_resnet', choices=['unet', 'vit_based', 'unet_resnet'], help='Model type to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    args = parser.parse_args()
    train_unet(args)
