import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from crack_seg import compute_dice, compute_iou, hybrid_loss
from finetune_sam3 import Sam3CrackDataset, forward_sam3, set_seed
from sam3.model_builder import build_sam3_image_model


class Sam3CrackDatasetWithNames(Sam3CrackDataset):
    """Extends the finetune dataset wrapper to also return filenames."""

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        fname = os.path.basename(self.img_paths[idx])
        return image, mask, fname


def make_loader(
    dataset_root: str,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = Sam3CrackDatasetWithNames(dataset_root, split, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def run_split_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    prompt: str,
    device: torch.device,
    split_name: str,
    max_steps: Optional[int] = None,
    save_dir: Optional[Path] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    loop = tqdm(loader, desc=f"Inference [{split_name}]")

    for step, batch in enumerate(loop):
        images, masks, names = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = forward_sam3(model, images, prompt, device)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = hybrid_loss(logits, masks)
        dice = compute_dice(logits, masks)
        iou = compute_iou(logits, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        num_batches += 1
        loop.set_postfix(loss=total_loss / num_batches, dice=total_dice / num_batches)

        if save_dir is not None:
            probs = torch.sigmoid(logits)
            for prob, name in zip(probs, names):
                mask_np = (prob.squeeze(0).cpu().numpy() > threshold).astype(np.uint8)
                Image.fromarray(mask_np * 255).save(save_dir / name)

        if max_steps is not None and (step + 1) >= max_steps:
            break

    if num_batches == 0:
        return {"loss": 0.0, "iou": 0.0, "dice": 0.0}

    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference-only SAM3 evaluation on OmniCrack30K."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/omnicrack30k",
        help="Root folder containing OmniCrack30K images/ and annotations/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["training", "validation", "test"],
        help="Dataset splits to evaluate.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--img_size", type=int, default=1008, help="Resize resolution.")
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
        help="Text prompt supplied to SAM3 during inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device identifier.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Limit batches processed per split (useful for smoke tests).",
    )
    parser.add_argument(
        "--save_masks_dir",
        type=str,
        default=None,
        help="If set, writes binary predictions for each split to this folder.",
    )
    parser.add_argument(
        "--binary_threshold",
        type=float,
        default=0.5,
        help="Threshold applied on sigmoid probabilities when exporting masks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    model = build_sam3_image_model(
        device=device.type,
        eval_mode=True,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=args.checkpoint_path is None,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    model.to(device)
    model.eval()

    save_root: Optional[Path] = None
    if args.save_masks_dir is not None:
        save_root = Path(args.save_masks_dir)
        save_root.mkdir(parents=True, exist_ok=True)

    all_results: List[str] = []
    for split in args.splits:
        loader = make_loader(
            args.dataset_root, split, args.img_size, args.batch_size, args.num_workers
        )
        split_save_dir = None
        if save_root is not None:
            split_save_dir = save_root / split
            split_save_dir.mkdir(parents=True, exist_ok=True)

        metrics = run_split_inference(
            model,
            loader,
            args.text_prompt,
            device,
            split_name=split,
            max_steps=args.max_steps,
            save_dir=split_save_dir,
            threshold=args.binary_threshold,
        )
        summary = (
            f"[{split}] "
            f"loss={metrics['loss']:.4f} | "
            f"IoU={metrics['iou']:.4f} | "
            f"Dice={metrics['dice']:.4f}"
        )
        print(summary)
        all_results.append(summary)

    print("========== Inference Summary ==========")
    for line in all_results:
        print(line)
    print("=======================================")


if __name__ == "__main__":
    main(parse_args())
