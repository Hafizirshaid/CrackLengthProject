# infer_crack_len.py

import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

from train_eval_crack_len import make_loaders
from models.crack_len_model import create_crack_len_model


@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) Build test split
    _, _, test_loader = make_loaders(
        root=args.kaggle_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # 2) Load model
    model = create_crack_len_model(
        model_name=args.model,
        device=device,
        seg_ckpt_path="",      # segmentation not needed here
        pooled_hw=args.pooled_hw,
        hidden_dim=args.hidden_dim,
        use_probs=True,
    )

    print(f"[Model] Loading checkpoint from {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    results = []
    all_gt = []
    all_pred = []

    pbar = tqdm(test_loader, desc="Inference", ncols=120)
    for imgs, masks, lengths in pbar:
        imgs = imgs.to(device)
        lengths = lengths.to(device)

        seg_logits, len_pred = model(imgs)

        gt = lengths.view(-1).cpu().numpy()
        pred = len_pred.view(-1).cpu().numpy()

        all_gt.append(gt)
        all_pred.append(pred)

        # Save for JSON
        for g, p in zip(gt, pred):
            results.append({"gt": float(g), "pred": float(p)})

    # ---- Convert to arrays for metrics ----
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    # ---- Metrics ----
    rmse = float(np.sqrt(np.mean((all_pred - all_gt) ** 2)))
    mae = float(np.mean(np.abs(all_pred - all_gt)))

    print(f"\n=== Test Metrics ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

    # ---- Save JSON ----
    out_path = args.out_json
    output = {
        "rmse": rmse,
        "mae": mae,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Done] Saved {len(results)} predictions + metrics → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference for crack length model (JSON output)")

    # Data parameters
    parser.add_argument("--kaggle_root", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Model params
    parser.add_argument("--model", type=str, default="unet_resnet")
    parser.add_argument("--pooled_hw", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    # Checkpoint + output
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="results/crack_len_results.json")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
