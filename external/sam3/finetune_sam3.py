import argparse
import os
import sys
from contextlib import nullcontext
from glob import glob
from typing import List, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from crack_seg import compute_dice, compute_iou, dice_loss, hybrid_loss

from sam3.model.data_misc import FindStage
from sam3.model_builder import build_sam3_image_model


def set_seed(seed: int) -> None:
    """Best-effort reproducibility helper."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Sam3CrackDataset(Dataset):
    """
    Minimal dataset wrapper that mirrors OmniCrackDataset but uses
    SAM3-friendly preprocessing (resize to model resolution and
    normalize to [-1, 1]).
    """

    def __init__(self, root_dir: str, split: str, img_size: int = 1008):
        self.split = split
        img_dir = os.path.join(root_dir, "images", split)
        ann_dir = os.path.join(root_dir, "annotations", split)
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        if not self.img_paths:
            raise RuntimeError(f"No images found in {img_dir}")
        self.ann_dir = ann_dir
        self.img_tf = T.Compose(
            [
                T.ToImage(),
                T.Resize((img_size, img_size)),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.mask_tf = T.Compose(
            [
                T.ToImage(),
                T.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.NEAREST
                ),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        fname = os.path.basename(img_path)
        ann_path = os.path.join(self.ann_dir, fname)
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Missing annotation for {img_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L")

        image = self.img_tf(image)
        mask = self.mask_tf(mask)
        # cracks are dark pixels -> treat < 0.5 as foreground
        mask = (mask < 0.5).float().squeeze(0)
        return image, mask


def forward_sam3(
    model: torch.nn.Module,
    images: torch.Tensor,
    prompt: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Runs SAM3 on a batch of images with the same text prompt and returns
    a (B, 1, H, W) tensor of combined mask logits.
    """
    batch_size = images.size(0)
    backbone_out = model.backbone.forward_image(images)
    text_outputs = model.backbone.forward_text([prompt] * batch_size, device=device)
    backbone_out.update(text_outputs)

    find_stage = FindStage(
        img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        text_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
        object_ids=None,
    )
    geometric_prompt = model._get_dummy_prompt(num_prompts=batch_size)

    # Temporarily switch off training flag to avoid matcher logic that
    # requires full SAM3 training targets. Submodules stay in training
    # mode since we only flip the flag on the parent module.
    was_training = model.training
    model.training = False
    outputs = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_stage,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )
    model.training = was_training

    mask_logits = outputs["pred_masks"]
    batch = mask_logits.shape[0]
    spatial_h, spatial_w = mask_logits.shape[-2], mask_logits.shape[-1]
    mask_logits = mask_logits.reshape(batch, -1, spatial_h, spatial_w)

    class_logits = outputs["pred_logits"].reshape(batch, -1)
    if "presence_logit_dec" in outputs:
        presence = torch.sigmoid(outputs["presence_logit_dec"]).reshape(batch, -1)
    else:
        presence = torch.ones_like(class_logits)
    scores = torch.sigmoid(class_logits) * presence
    weights = scores / (scores.sum(dim=1, keepdim=True) + 1e-6)
    weights = weights.unsqueeze(-1).unsqueeze(-1)
    combined_logits = (mask_logits * weights).sum(dim=1, keepdim=True)
    return combined_logits


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    prompt: str,
    device: torch.device,
    max_steps: Optional[int],
    use_tqdm: bool = False,
    desc: Optional[str] = None,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    count = 0
    iterator = loader
    if use_tqdm:
        iterator = tqdm(loader, desc=desc)
    with torch.no_grad():
        for step, (images, masks) in enumerate(iterator):
            images = images.to(device)
            masks = masks.to(device)
            logits = forward_sam3(model, images, prompt, device)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = hybrid_loss(logits, masks)
            total_loss += loss.item()
            total_iou += compute_iou(logits.detach(), masks)
            total_dice += compute_dice(logits.detach(), masks)
            count += 1
            if max_steps is not None and (step + 1) >= max_steps:
                break
    if count == 0:
        return 0.0, 0.0, 0.0
    return total_loss / count, total_iou / count, total_dice / count


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, "sam3_finetuned_best.pt")
    final_ckpt_path = os.path.join(args.output_dir, "sam3_finetuned_last.pt")
    curve_path = os.path.join(args.output_dir, "sam3_finetune_loss_curve.png")

    model = build_sam3_image_model(
        device=device.type,
        eval_mode=False,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=args.checkpoint_path is None,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    model.to(device)
    model.train()

    train_ds = Sam3CrackDataset(args.dataset_root, "training", img_size=args.img_size)
    val_ds = Sam3CrackDataset(args.dataset_root, "validation", img_size=args.img_size)
    test_ds = Sam3CrackDataset(args.dataset_root, "test", img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (images, masks) in enumerate(loop):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.amp.autocast(device_type="cuda")
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                logits = forward_sam3(model, images, args.text_prompt, device)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = hybrid_loss(logits, masks)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            steps += 1
            loop.set_postfix(train_loss=running_loss / steps)
            if args.max_steps is not None and (step + 1) >= args.max_steps:
                break

        epoch_loss = running_loss / max(1, steps)
        train_losses.append(epoch_loss)

        val_loss, val_iou, val_dice = evaluate_split(
            model, val_loader, args.text_prompt, device, args.max_steps
        )
        val_losses.append(val_loss)
        print(
            f"[Epoch {epoch}] train_loss={epoch_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_iou={val_iou:.4f} | val_dice={val_dice:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> Saved new best checkpoint to {best_ckpt_path}")

    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Saved last checkpoint to {final_ckpt_path}")

    # Plot curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("SAM3 Finetuning Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {curve_path}")

    # Evaluate best checkpoint on the held-out test set
    if os.path.exists(best_ckpt_path):
        state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best checkpoint from {best_ckpt_path} for testing.")
    test_loss, test_iou, test_dice = evaluate_split(
        model, test_loader, args.text_prompt, device, args.max_steps
    )
    print("========== Test Results ==========")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU:  {test_iou:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print("==================================")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SAM3 on OmniCrack30K.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/omnicrack30k",
        help="Root folder containing images/ and annotations/.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Cap the number of steps per epoch for quick experiments.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--img_size", type=int, default=1008, help="Image resolution fed to SAM3.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a local SAM3 checkpoint (.pt). Leave empty to download from HF.",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="crack",
        help="Text prompt supplied to SAM3 during training/inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device identifier.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sam3_finetune",
        help="Where to store checkpoints and plots.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable mixed precision when training on CUDA.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
