#!/usr/bin/env python
"""
cl_eval_sam3.py

Compute centerline IoU (clIoU) for precomputed SAM3 masks.

Conventions follow evaluation.py:

- Ground truth centerlines:
    * crack = 0 (black), background = 255 (white)
    * we use (true_img == 0) → 1 for crack
- Predicted SAM3 masks (saved by inference_sam3.py):
    * crack = 255 (white), background = 0 (black)
    * we threshold > 128 to get 1 for crack
    * then run skeletonization + invert exactly like evaluation.py
"""

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from skimage.morphology import thin
from sklearn.metrics import jaccard_score  # used only for sanity if you want

# Import subset list + tolerance function from original evaluation
from evaluation import SUBSETS, apply_tolerance  # type: ignore


def run_sam3_cliou(
    datapath: Path,
    pred_root: Path,
    texpath: str,
    split: str,
    subset: str | None,
    tolerance: int,
):
    """
    Compute clIoU for SAM3 predictions stored as binary masks.

    Parameters
    ----------
    datapath : Path
        Root of omnicrack30k dataset (same as in evaluation.py).
    pred_root : Path
        Root directory where SAM masks are saved.
        Expected layout: pred_root / split / <filename>.png
    texpath : str
        Path to save LaTeX string.
    split : {"test", "validation"}
        Dataset split.
    subset : str | None
        If given, only evaluate that subset (e.g., "AEL").
        Otherwise, iterate over SUBSETS[split].
    tolerance : int
        Tolerance used in clIoU (same as evaluation.py).
    """

    print("=== SAM3 clIoU evaluation ===")
    print(f"GT root:    {datapath}")
    print(f"PRED root:  {pred_root}")
    print(f"Split:      {split}")
    print(f"Subset:     {subset if subset is not None else 'ALL'}")
    print(f"Tolerance:  {tolerance}")

    subsets = [subset] if subset is not None else SUBSETS[split]

    # Instead of np.append for every image, track IoU counts per subset
    inter_counts = {key: 0 for key in subsets}
    union_counts = {key: 0 for key in subsets}

    tex = ""

    for key in subsets:
        img_dir = datapath / "images" / split
        centerline_dir = datapath / "centerlines" / split
        pred_dir = pred_root / split

        img_paths = sorted(img_dir.glob(f"{key}*.png"))

        if not img_paths:
            print(f"[WARN] No images for subset {key} in split {split}")
            continue

        print(f"\n[Subset] {key}  (images: {len(img_paths)})")

        for img_path in tqdm(img_paths, desc=f"{key} [{split}]"):
            # --- Load GT centerline image ---
            true_path = centerline_dir / img_path.name
            if not true_path.exists():
                raise FileNotFoundError(f"Missing GT centerline: {true_path}")

            true_img = cv2.imread(str(true_path), cv2.IMREAD_GRAYSCALE)

            # GT: crack = 0 (black) → 1 for crack
            true_bin = np.uint8(true_img == 0)

            # --- Load SAM3 predicted mask ---
            pred_path = pred_dir / img_path.name
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing SAM3 prediction: {pred_path}")

            pred_mask_gray = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

            # SAM saved as: crack = 255 (white), bg = 0 (black)
            # → 1 for crack via > 128
            pred_bin = (pred_mask_gray > 128).astype(np.uint8)

            # --- Compute centerlines, mirroring evaluation.py ---
            # thin expects binary (0/1) mask with foreground = 1
            centerlines = np.uint8(255 * thin(pred_bin))

            # evaluation.py inverts for visualization and then uses "== 0" as crack
            centerlines = 255 - centerlines  # now crack lines are 0

            pred_bin_center = np.uint8(centerlines == 0)

            # --- Make sure shapes match (e.g., if SAM ran at different resolution) ---
            if pred_bin_center.shape != true_bin.shape:
                pred_bin_center = cv2.resize(
                    pred_bin_center,
                    (true_bin.shape[1], true_bin.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # --- Apply tolerance (same function as evaluation.py) ---
            true_tol, pred_tol = apply_tolerance(true_bin, pred_bin_center, tol=tolerance)

            # --- Flatten and remove true negatives ---
            true_flat = true_tol.flatten().astype(bool)
            pred_flat = pred_tol.flatten().astype(bool)

            if true_flat.size == 0:
                continue

            keep = np.logical_or(true_flat, pred_flat)
            true_fg = true_flat[keep]
            pred_fg = pred_flat[keep]

            if true_fg.size == 0:
                continue

            # --- Accumulate intersection and union for this subset ---
            inter = np.logical_and(true_fg, pred_fg).sum()
            union = np.logical_or(true_fg, pred_fg).sum()

            inter_counts[key] += int(inter)
            union_counts[key] += int(union)

        # Compute clIoU for this subset
        if union_counts[key] == 0:
            cliou = 0.0
            print(f"[Result] {key}: no foreground pixels; clIoU set to 0.0")
        else:
            cliou = inter_counts[key] / union_counts[key]
            print(f"[Result] {key}: clIoU = {cliou:.3f}")

        # LaTeX string like evaluation.py ("& xx.x ")
        tex += f"& {100 * cliou:.1f} "

    if texpath is not None:
        out_path = Path(texpath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"\nWrote LaTeX results to: {texpath}")


def main():
    parser = ArgumentParser(description="Centerline IoU (clIoU) eval for SAM3 masks.")
    parser.add_argument(
        "split",
        nargs="?",
        default="test",
        choices=["test", "validation"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=str,
        default=None,
        help="Subset to evaluate (e.g., 'AEL'). If omitted, evaluate all in SUBSETS[split].",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=int,
        default=4,
        help="Tolerance for clIoU (same as evaluation.py).",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/mnt/home/irshaid2/crack_seg/omnicrack30k",
        help="Path to root folder of omnicrack30k dataset.",
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default="results/sam3_inference",
        help="Root dir where SAM3 masks are stored (same as --save_masks_dir in inference_sam3.py).",
    )
    parser.add_argument(
        "-tp",
        "--texpath",
        type=str,
        default="results_sam3_cliou.tex",
        help="Path to save LaTeX table snippet.",
    )

    args = parser.parse_args()

    run_sam3_cliou(
        datapath=Path(args.datapath),
        pred_root=Path(args.pred_root),
        texpath=args.texpath,
        split=args.split,
        subset=args.subset,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
