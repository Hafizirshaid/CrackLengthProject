#!/usr/bin/env python3
"""Utility to run SAM3 image inference on a small batch (default: 10 images)."""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 inference on (up to) 10 images using the text prompt interface."
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="dataset/omnicrack30k/images/test",
        help="Folder that contains the images you want to segment.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="*.png",
        help="Glob pattern (relative to image_root) that selects the images.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Process only the first N images that match the glob pattern.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sam3_batch10",
        help="Directory where binary masks will be written.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="crack",
        help="Text prompt supplied to SAM3.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a local SAM3 checkpoint (.pt). "
        "Leave empty to pull the default weights from Hugging Face.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1008,
        help="Resolution used by the processor (keep in sync with training).",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Masks with scores below this value will be discarded.",
    )
    return parser.parse_args()


def collect_images(root: Path, glob_pattern: str, limit: int) -> List[Path]:
    candidates = [
        path
        for path in sorted(root.glob(glob_pattern))
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No images found inside '{root}' that match '{glob_pattern}'."
        )
    return candidates[:limit]


def load_model(device: torch.device, checkpoint_path: Optional[str]) -> torch.nn.Module:
    model = build_sam3_image_model(
        device=device.type,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    model.to(device)
    model.eval()
    return model


def save_top_mask(state: dict, output_path: Path) -> None:
    scores = state.get("scores")
    masks = state.get("masks")
    if scores is None or masks is None or scores.numel() == 0:
        print(f"[WARN] No confident masks for {output_path.stem}, skipping.")
        return

    best_idx = int(torch.argmax(scores).item())
    best_mask = masks[best_idx].squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(best_mask, mode="L").save(output_path)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    image_root = Path(args.image_root)
    output_root = Path(args.output_dir)

    image_paths = collect_images(image_root, args.glob_pattern, args.num_images)
    model = load_model(device, args.checkpoint_path)
    processor = Sam3Processor(
        model,
        resolution=args.img_size,
        device=device.type,
        confidence_threshold=args.confidence_threshold,
    )

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=args.prompt, state=state)

        output_path = output_root / f"{img_path.stem}_mask.png"
        save_top_mask(state, output_path)
        score_str = (
            f"{float(state['scores'].max()):.3f}"
            if state.get("scores") is not None and state["scores"].numel() > 0
            else "n/a"
        )
        print(f"[OK] {img_path.name} -> {output_path} (score={score_str})")


if __name__ == "__main__":
    run(parse_args())
