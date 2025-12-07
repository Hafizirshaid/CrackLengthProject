# train_eval_crack_len.py

import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler


from dataset import KaggleCrackLenDataset 

# seg losses/metrics from existing crack_seg.py
from crack_seg import hybrid_loss, compute_iou, compute_dice

# model & seg backbone builder
from models.crack_len_model import create_crack_len_model

from tqdm import tqdm
import matplotlib.pyplot as plt

def make_loaders(
    root: str,
    img_size: int,
    batch_size: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Build train/val/test dataloaders by random splitting the Kaggle dataset.
    Assumes KaggleCrackLenDataset(root, img_size) covers the whole Kaggle set.
    """
    full_ds = KaggleCrackLenDataset(root=root, img_size=img_size)

    #print('Total dataset size:', len(full_ds))
    n = len(full_ds)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    n_train = n - n_val - n_test

    #print('n train:', n_train, 'n val:', n_val, 'n test:', n_test)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(
        full_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        full_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        full_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def _aggregate_init():
    return {
        "seg_loss": 0.0,
        "len_loss": 0.0,
        "total_loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "len_mse": 0.0,
        "len_mae": 0.0,
        "n_batches": 0,
    }


def _aggregate_update(agg, seg_logits, masks, len_pred, lengths,
                      seg_loss, len_loss, total_loss):
    agg["seg_loss"] += seg_loss.item()
    agg["len_loss"] += len_loss.item()
    agg["total_loss"] += total_loss.item()
    agg["iou"] += compute_iou(seg_logits, masks)
    agg["dice"] += compute_dice(seg_logits, masks)

    mse = F.mse_loss(len_pred, lengths, reduction="mean").item()
    mae = F.l1_loss(len_pred, lengths, reduction="mean").item()
    agg["len_mse"] += mse
    agg["len_mae"] += mae
    agg["n_batches"] += 1


def _aggregate_finalize(agg):
    n = max(agg["n_batches"], 1)
    return {
        "seg_loss": agg["seg_loss"] / n,
        "len_loss": agg["len_loss"] / n,
        "total_loss": agg["total_loss"] / n,
        "iou": agg["iou"] / n,
        "dice": agg["dice"] / n,
        "len_mse": agg["len_mse"] / n,
        "len_rmse": (agg["len_mse"] / n) ** 0.5,
        "len_mae": agg["len_mae"] / n,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    train_mode: str,
    alpha_len: float,
):
    model.train()
    agg = _aggregate_init()

    pbar = tqdm(loader, desc="Training", ncols=120)

    for imgs, masks, lengths in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        seg_logits, len_pred = model(imgs)

        len_loss = F.mse_loss(len_pred, lengths)

        if train_mode == "joint":
            seg_loss = hybrid_loss(seg_logits, masks)
            total_loss = seg_loss + alpha_len * len_loss
        else:
            seg_loss = torch.tensor(0.0, device=device)
            total_loss = len_loss

        total_loss.backward()
        optimizer.step()

        _aggregate_update(agg, seg_logits, masks, len_pred, lengths,
                          seg_loss, len_loss, total_loss)

        # update progress bar text
        pbar.set_postfix({
            "tot": f"{total_loss.item():.4f}",
            "len": f"{len_loss.item():.4f}"
        })

    return _aggregate_finalize(agg)


@torch.no_grad()
@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device,
    train_mode: str,
    alpha_len: float,
):
    model.eval()
    agg = _aggregate_init()

    pbar = tqdm(loader, desc="Validating", ncols=120)

    for imgs, masks, lengths in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        lengths = lengths.to(device)

        seg_logits, len_pred = model(imgs)
        len_loss = F.mse_loss(len_pred, lengths)

        if train_mode == "joint":
            seg_loss = hybrid_loss(seg_logits, masks)
            total_loss = seg_loss + alpha_len * len_loss
        else:
            seg_loss = torch.tensor(0.0, device=device)
            total_loss = len_loss

        _aggregate_update(agg, seg_logits, masks, len_pred, lengths,
                          seg_loss, len_loss, total_loss)

        pbar.set_postfix({
            "tot": f"{total_loss.item():.4f}",
            "len": f"{len_loss.item():.4f}"
        })

    return _aggregate_finalize(agg)



def main():
    parser = argparse.ArgumentParser(
        description="Train/eval crack length model on Kaggle dataset"
    )

    parser.add_argument(
        "--kaggle_root",
        type=str,
        required=True,
        help="Path to crack_final directory (that has 1-Segmentation, 4-Crack Length)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet_resnet",
        choices=["unet", "vit_based", "unet_resnet"],
        help="Segmentation backbone to wrap",
    )
    parser.add_argument(
        "--seg_ckpt",
        type=str,
        default="",
        help="Optional seg-only checkpoint (pretrained on OmniCrack30K, etc.)",
    )

    parser.add_argument(
        "--train_mode",
        type=str,
        default="freeze_seg",
        choices=["freeze_seg", "joint", "len_only"],
        help="Training mode: freeze_seg / joint / len_only",
    )
    parser.add_argument("--alpha_len", type=float, default=1.0,
                        help="Weight for length loss in joint mode")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--pooled_hw", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) data
    train_loader, val_loader, test_loader = make_loaders(
        root=args.kaggle_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # 2) model
    model = create_crack_len_model(
        model_name=args.model,
        device=device,
        seg_ckpt_path=args.seg_ckpt,
        pooled_hw=args.pooled_hw,
        hidden_dim=args.hidden_dim,
        use_probs=True,
    )

    # freeze seg backbone if needed
    if args.train_mode == "freeze_seg":
        print(">>> Freezing segmentation backbone parameters.")
        for p in model.seg_model.parameters():
            p.requires_grad = False
    else:
        print(">>> Training segmentation backbone end-to-end.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 3) train
    run_name = f"{args.model}_{args.train_mode}_lr{args.lr}_bs{args.batch_size}"
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt = os.path.join("checkpoints", f"{run_name}_best_len_rmse.pth")
    best_val_rmse = float("inf")

    train_curve = []
    val_curve = []


    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_mode=args.train_mode,
            alpha_len=args.alpha_len,
        )
        val_stats = eval_one_epoch(
            model,
            val_loader,
            device,
            train_mode=args.train_mode,
            alpha_len=args.alpha_len,
        )

        print("[Train] "
              f"tot={train_stats['total_loss']:.4f} | "
              f"seg={train_stats['seg_loss']:.4f} | "
              f"len={train_stats['len_loss']:.4f} | "
              f"len_rmse={train_stats['len_rmse']:.4f} | "
              f"len_mae={train_stats['len_mae']:.4f} | "
              f"IoU={train_stats['iou']:.4f} | "
              f"Dice={train_stats['dice']:.4f}")
        print("[Val]   "
              f"tot={val_stats['total_loss']:.4f} | "
              f"seg={val_stats['seg_loss']:.4f} | "
              f"len={val_stats['len_loss']:.4f} | "
              f"len_rmse={val_stats['len_rmse']:.4f} | "
              f"len_mae={val_stats['len_mae']:.4f} | "
              f"IoU={val_stats['iou']:.4f} | "
              f"Dice={val_stats['dice']:.4f}")

        if val_stats["len_rmse"] < best_val_rmse:
            best_val_rmse = val_stats["len_rmse"]
            torch.save(model.state_dict(), best_ckpt)
            print(f">>> New best val len RMSE {best_val_rmse:.4f}, saved to {best_ckpt}")

        train_curve.append(train_stats["total_loss"])
        val_curve.append(val_stats["total_loss"])
    print("\n[Done] Best val len RMSE:", best_val_rmse)

    plt.figure(figsize=(8,5))
    plt.plot(train_curve, label="Train Total Loss", marker="o")
    plt.plot(val_curve, label="Val Total Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Crack Length Model Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    curve_path = f"plots/{run_name}_loss_curve.png"
    plt.savefig(curve_path)
    print(f"Saved loss curve → {curve_path}")


    # 4) test best model
    print("\n[Eval] Loading best checkpoint for test...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_stats = eval_one_epoch(
        model,
        test_loader,
        device,
        train_mode=args.train_mode,
        alpha_len=args.alpha_len,
    )
    print("[Test]  "
          f"tot={test_stats['total_loss']:.4f} | "
          f"seg={test_stats['seg_loss']:.4f} | "
          f"len={test_stats['len_loss']:.4f} | "
          f"len_rmse={test_stats['len_rmse']:.4f} | "
          f"len_mae={test_stats['len_mae']:.4f} | "
          f"IoU={test_stats['iou']:.4f} | "
          f"Dice={test_stats['dice']:.4f}")


if __name__ == "__main__":
    main()
