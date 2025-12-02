import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from inference import save_segmentation_visualization
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_binary_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask_np = (np.array(mask) < 128).astype(np.uint8)
    return mask_np


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice_den = pred.sum() + gt.sum()
    dice = (2 * inter) / (dice_den + 1e-6) if dice_den > 0 else 0.0
    iou = inter / (union + 1e-6) if union > 0 else 0.0
    return iou, dice


def infer_single_image(
    processor: Sam3Processor,
    image_path: Path,
    prompt: str,
    score_threshold: float,   # <-- ignored now
):
    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    # If no result at all
    if masks is None or masks.numel() == 0:
        empty = np.zeros((image.height, image.width), dtype=np.uint8)
        return empty, 0.0, image, None, None, None

    # Move tensors to CPU / float
    mask_tensor = masks.detach().cpu().float()
    score_tensor = scores.detach().cpu().float()

    # Normalize shape: (N, 1, H, W) → (N, H, W)
    if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
        mask_tensor = mask_tensor[:, 0]
    elif mask_tensor.ndim != 3:
        raise ValueError(f"Unexpected mask tensor shape: {mask_tensor.shape}")

    N = mask_tensor.shape[0]
    if N == 0:
        empty = np.zeros((image.height, image.width), dtype=np.uint8)
        return empty, 0.0, image, None, None, None

    # ----------- ⭐ NEW: sort by score, take top 10 masks ----------- #
    top_k = min(10, N)
    sorted_idx = torch.argsort(score_tensor, descending=True)
    idx_topk = sorted_idx[:top_k]

    mask_topk = mask_tensor[idx_topk]   # shape (top_k, H, W)
    score_topk = score_tensor[idx_topk]
    # ---------------------------------------------------------------- #

    # Combine top-k masks: simple OR
    combined = (mask_topk > 0.5).any(dim=0).to(torch.uint8).numpy()

    best_score = float(score_topk.max().item())

    return combined, best_score, image, masks, boxes, scores



def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def run_split(
    processor: Sam3Processor,
    split: str,
    dataset_root: Path,
    prompt: str,
    score_threshold: float,
    save_root: Optional[Path],
    max_images: Optional[int],
    viz_root: Optional[Path],
    viz_max_images: int,
) -> Tuple[float, float, float]:
    img_dir = dataset_root / "images" / split
    ann_dir = dataset_root / "annotations" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {ann_dir}")

    image_paths = list_images(img_dir)
    if max_images is not None:
        image_paths = image_paths[: max(0, max_images)]

    total_iou = 0.0
    total_dice = 0.0
    total_score = 0.0
    count = 0

    split_save_dir = save_root / split if save_root is not None else None
    if split_save_dir is not None:
        split_save_dir.mkdir(parents=True, exist_ok=True)

    split_viz_dir = viz_root / split if viz_root is not None else None
    if split_viz_dir is not None:
        split_viz_dir.mkdir(parents=True, exist_ok=True)
    viz_count = 0

    for img_path in tqdm(image_paths, desc=f"Inference [{split}]"):
        pred_mask, score, pil_image, _, _, _ = infer_single_image(
            processor, img_path, prompt, score_threshold
        )
        gt_mask_path = ann_dir / img_path.name
        if not gt_mask_path.exists():
            raise FileNotFoundError(f"Missing annotation for {img_path.name}")
        gt_mask = load_binary_mask(gt_mask_path)

        iou, dice = compute_metrics(pred_mask, gt_mask)
        total_iou += iou
        total_dice += dice
        total_score += score
        count += 1

        if split_save_dir is not None:
            save_binary_mask(pred_mask, split_save_dir / img_path.name)

        if (
            split_viz_dir is not None
            and viz_count < viz_max_images
        ):
            save_segmentation_visualization(
                pil_image,
                gt_mask,
                pred_mask,
                img_path.name,
                iou,
                dice,
                split_viz_dir,
            )
            viz_count += 1

    if count == 0:
        return 0.0, 0.0, 0.0

    return total_iou / count, total_dice / count, total_score / count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 inference over a dataset.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/omnicrack30k",
        help="Root folder containing images/ and annotations/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["training", "validation", "test"],
        help="Dataset splits to evaluate.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a SAM3 checkpoint (.pt). Leave empty to download from HF.",
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
        help="Device identifier (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1008,
        help="Resolution passed to Sam3Processor.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Discard predictions whose best mask score falls below this value.",
    )
    parser.add_argument(
        "--save_masks_dir",
        type=str,
        default="results/sam3_inference",
        help="Directory where predicted masks are stored.",
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default=None,
        help="Optional directory to store RGB/GT/pred visualizations.",
    )
    parser.add_argument(
        "--viz_max_images",
        type=int,
        default=16,
        help="Maximum number of visualizations saved per split.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap on number of images processed per split.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    device_str = device.type

    model = build_sam3_image_model(
        device=device_str,
        eval_mode=True,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=args.checkpoint_path is None,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    model.to(device)
    model.eval()

    processor = Sam3Processor(
        model,
        resolution=args.img_size,
        device=device_str,
        confidence_threshold=args.score_threshold,
    )

    dataset_root = Path(args.dataset_root)
    save_root = Path(args.save_masks_dir) if args.save_masks_dir is not None else None
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)
    viz_root = Path(args.viz_dir) if args.viz_dir is not None else None
    if viz_root is not None:
        viz_root.mkdir(parents=True, exist_ok=True)

    summaries: List[str] = []
    for split in args.splits:
        avg_iou, avg_dice, avg_score = run_split(
            processor,
            split,
            dataset_root,
            args.text_prompt,
            args.score_threshold,
            save_root,
            args.max_images,
            viz_root,
            args.viz_max_images,
        )
        summary = (
            f"[{split}] IoU={avg_iou:.4f} | Dice={avg_dice:.4f} | "
            f"AvgScore={avg_score:.4f}"
        )
        print(summary)
        summaries.append(summary)

    print("========== Inference Summary ==========")
    for line in summaries:
        print(line)
    print("=======================================")


if __name__ == "__main__":
    main(parse_args())
