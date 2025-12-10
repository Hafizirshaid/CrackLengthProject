import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from models.unet_resnet import get_unet_resnet50

from crack_seg import compute_iou, compute_dice, hybrid_loss
import argparse
from models.vit_based import CrackSegMixtureModel as ViTBasedSegModel
from models.unet import UNet


# DATASETS = ['AEL', 'BCL', 'Ceramic', 'CFD', 'CRACK500', 'CrackLS315']
# DATASETS = ['CRKWH100', 'CrSpEE', 'CSSC', 'DeepCrack', 'DIC', 'GAPS384']
DATASETS = ['Khanh11k', 'LCW', 'Masonry', 'S2DS', 'Stone331', 'TopoDS', 'UAV75']
# , 'Khanh11k', 'LCW', 'Masonry', 'S2DS', 'Stone331', 'TopoDS', 'UAV75']
# ============================================================
# INFERENCE ON TEST SET
# ============================================================
def evaluate_test_set(model_path, root_dir, img_size=256, save_folder="inference_results"):
    """
    Evaluates U-Net on the test dataset using IoU and Dice.
    Assumes:
        root_dir/images/test/*.png
        root_dir/annotations/test/*.png
    """
    device = (
         "cpu"
    )

    # Create output directory for plots
    plot_dir = os.path.join(root_dir, f"{save_folder}_results_plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Using device: {device}")
    if args.model == 'unet':
        model = UNet().to(device)
    elif args.model == 'vit_based':
        model = ViTBasedSegModel().to(device)
    elif args.model == "unet_resnet":
        model = get_unet_resnet50(num_classes=1, pretrained=True, freeze_encoder=False, bilinear=True).to(device)
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
                       interpolation=Image.NEAREST)

    # Directories
    test_img_dir = os.path.join(root_dir, "images", "test")
    test_ann_dir = os.path.join(root_dir, "annotations", "test")
    test_centerline_dir = os.path.join(root_dir, "centerlines", "test")


    for dataset in DATASETS:

        # filter only LCW_ images
        test_imgs = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png") and f.startswith(dataset)])
        # test_imgs = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])

        total_iou = 0
        total_dice = 0
        total_loss = 0

        print(f"Evaluating {len(test_imgs)} test images...")
        count = 0
    
        for fname in tqdm(test_imgs):
            img_path = os.path.join(test_img_dir, fname)
            ann_path = os.path.join(test_ann_dir, fname)
            centerline_path = os.path.join(test_centerline_dir, fname)


            

            # Load original image + mask for visualization
            original_img = Image.open(img_path).convert("RGB")
            # original_mask = Image.open(ann_path).convert("L")
            original_img = original_img.resize((img_size, img_size), Image.BILINEAR)

            original_img.save(os.path.join(plot_dir, f"{dataset}_original_img.png"))

            # # Load and preprocess image + mask for inference
            img = img_tf(original_img).unsqueeze(0).to(device)
            # --------- Load mask (correct) ---------
            original_mask = Image.open(ann_path).convert("L")
            original_mask = original_mask.resize((img_size, img_size), Image.NEAREST)
            original_mask.save(os.path.join(plot_dir, f"{dataset}_original_mask.png"))

            mask_np = np.array(original_mask)

            # cracks are dark (near 0), background is light (near 255)
            # treat dark pixels as 1 (crack), light pixels as 0
            mask_np = (mask_np < 128).astype(np.uint8)    # <-- key change

            mask = Image.fromarray(mask_np)
            mask = mask.resize((img_size, img_size), Image.NEAREST)

            mask = torch.from_numpy(np.array(mask)).float()   # (H,W)
            mask = mask.unsqueeze(0)   


            centerline = Image.open(centerline_path).convert("L")
            centerline = centerline.resize((img_size, img_size), Image.NEAREST)
            centerline = centerline.resize((img_size, img_size), Image.NEAREST)

            centerline.save(os.path.join(plot_dir, f"{dataset}_centerline.png"))

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
                axes[0].set_title(f'Input Image')
                axes[0].axis('off')
                
                # Ground truth mask
                axes[1].imshow(np.array(mask_np), cmap='gray')
                axes[1].set_title(f'Ground Truth\nMask')
                axes[1].axis('off')
                
                # Prediction
                pred_np = pred_mask.squeeze().cpu().numpy()

                # resize pred_np to match original_mask size
                original_mask_size = original_mask.size  # (width, height)
                pred_resized = np.array(Image.fromarray((pred_np * 255).astype(np.uint8)).resize(original_mask_size, Image.NEAREST)) / 255.0
                

                # do 182 thing here
                # pred_resized_tosave = (pred_resized > 128).astype(np.uint8)
                pred_resized_tosave = (pred_resized > 0.5).astype(np.uint8)
                # If prediction colors are inverted compared to ground truth, flip them
                pred_resized_tosave = 1 - pred_resized_tosave
                pred_img = Image.fromarray((pred_resized_tosave * 255).astype(np.uint8))
                pred_img.save(os.path.join(plot_dir, f"{dataset}_{args.model}_pred_mask.png"))

                axes[2].imshow(pred_resized, cmap='gray')
                # show IoU and Dice in the prediction plot title (as percentages)
                axes[2].set_title(f'Prediction Mask\nIoU: {iou*100:.1f}%  Dice: {dice*100:.1f}%')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(plot_dir, f"inference_{fname.replace('.png', '.png')}")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()  # Close to free memory

            # We only need the first image. 
            break
        # N = len(test_imgs)
        N = 10
        print("\n======= TEST RESULTS =======")
        print(f"Test Loss: {total_loss / N:.4f}")
        print(f"Test IoU : {total_iou / N:.4f}")
        print(f"Test Dice: {total_dice / N:.4f}")
        print(f"Plots saved to: {plot_dir}")
        print("============================\n")

        # return total_loss / N, total_iou / N, total_dice / N



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="U-Net Inference on Test Set")
    parser.add_argument('--model_path', type=str, default='checkpoints/unet_best.pth', help='Path to the trained U-Net model')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'vit_based', 'unet_resnet'], help='Model type to use')
    parser.add_argument('--save_folder', type=str, default='inference_results', help='Folder to save inference results')
    args = parser.parse_args()
    
    # ------ TEST ------
    evaluate_test_set(
        model_path=args.model_path,
        root_dir="/mnt/home/irshaid2/crack_seg/omnicrack30k",
        img_size=256,
        save_folder=args.save_folder
    )