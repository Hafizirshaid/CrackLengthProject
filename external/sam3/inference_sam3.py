import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT))
from finetune_sam3 import (
    Sam3CrackDataset,
    evaluate_split,
    forward_sam3,
    set_seed,
)
from inference import save_segmentation_visualization
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
    return_names: bool = False,
) -> DataLoader:
    dataset_cls = Sam3CrackDatasetWithNames if return_names else Sam3CrackDataset
    dataset = dataset_cls(dataset_root, split, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def save_split_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    prompt: str,
    device: torch.device,
    split_name: str,
    save_dir: Optional[Path],
    max_steps: Optional[int] = None,
    threshold: float = 0.5,
    dataset_root: Optional[Path] = None,
    viz_dir: Optional[Path] = None,
    viz_max_samples: int = 16,
) -> None:
    if save_dir is None and viz_dir is None:
        return
    model.eval()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    loop = tqdm(loader, desc=f"Exporting masks [{split_name}]")
    viz_count = 0
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

        probs = torch.sigmoid(logits)
        bin_preds = (probs > threshold).float()
        for idx_in_batch, (prob, name) in enumerate(zip(bin_preds, names)):
            mask_np = prob.squeeze(0).cpu().numpy().astype(np.uint8)
            if save_dir is not None:
                Image.fromarray(mask_np * 255).save(save_dir / name)

            if (
                viz_dir is not None
                and dataset_root is not None
                and viz_count < viz_max_samples
            ):
                img_path = dataset_root / "images" / split_name / name
                ann_path = dataset_root / "annotations" / split_name / name
                if not img_path.exists() or not ann_path.exists():
                    continue

                original_img = Image.open(img_path).convert("RGB")
                gt_mask = Image.open(ann_path).convert("L")
                gt_mask_np = (np.array(gt_mask) < 128).astype(np.uint8)

                pred_resized = (
                    np.array(
                        Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                            original_img.size, Image.NEAREST
                        )
                    )
                    / 255.0
                )
                pred_binary = (pred_resized > 0.5).astype(np.uint8)
                inter = np.logical_and(pred_binary, gt_mask_np).sum()
                union = np.logical_or(pred_binary, gt_mask_np).sum()
                iou = float(inter / (union + 1e-6)) if union > 0 else 0.0
                denom = pred_binary.sum() + gt_mask_np.sum()
                dice = (
                    float((2 * inter) / (denom + 1e-6)) if denom > 0 else 0.0
                )

                save_segmentation_visualization(
                    original_img,
                    gt_mask_np,
                    pred_resized,
                    name,
                    iou,
                    dice,
                    viz_dir,
                )
                viz_count += 1

        if max_steps is not None and (step + 1) >= max_steps:
            break


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
        "--viz_dir",
        type=str,
        default="viz_sam3",
        help="Optional directory for triptych visualizations (image/GT/pred).",
    )
    parser.add_argument(
        "--viz_max_images",
        type=int,
        default=16,
        help="Maximum number of visualizations to save per split.",
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
    dataset_root_path = Path(args.dataset_root)

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

    viz_root: Optional[Path] = None
    if args.viz_dir is not None:
        viz_root = Path(args.viz_dir)
        viz_root.mkdir(parents=True, exist_ok=True)

    all_results: List[str] = []
    for split in args.splits:
        metrics_loader = make_loader(
            str(dataset_root_path),
            split,
            args.img_size,
            args.batch_size,
            args.num_workers,
            return_names=False,
        )
        metrics = evaluate_split(
            model,
            metrics_loader,
            args.text_prompt,
            device,
            args.max_steps,
        )

        need_exports = (save_root is not None) or (viz_root is not None)
        if need_exports:
            export_loader = make_loader(
                str(dataset_root_path),
                split,
                args.img_size,
                args.batch_size,
                args.num_workers,
                return_names=True,
            )
            split_save_dir = None
            split_viz_dir = None
            if save_root is not None:
                split_save_dir = save_root / split
                split_save_dir.mkdir(parents=True, exist_ok=True)
            if viz_root is not None:
                split_viz_dir = viz_root / split
                split_viz_dir.mkdir(parents=True, exist_ok=True)
            save_split_predictions(
                model,
                export_loader,
                args.text_prompt,
                device,
                split_name=split,
                save_dir=split_save_dir,
                max_steps=args.max_steps,
                threshold=args.binary_threshold,
                dataset_root=dataset_root_path,
                viz_dir=split_viz_dir,
                viz_max_samples=args.viz_max_images,
            )

        summary = (
            f"[{split}] "
            f"loss={metrics[0]:.4f} | "
            f"IoU={metrics[1]:.4f} | "
            f"Dice={metrics[2]:.4f}"
        )
        print(summary)
        all_results.append(summary)

    print("========== Inference Summary ==========")
    for line in all_results:
        print(line)
    print("=======================================")


if __name__ == "__main__":
    main(parse_args())
