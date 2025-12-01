import os
import sys
import argparse
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T

# make the external codes package importable
EXTERNAL_CODES = os.path.join(os.path.dirname(__file__), 'external', 'Omaralmaqtari-Efficient-Multiscale-Transformer-2785c26', 'codes')
if EXTERNAL_CODES not in sys.path:
    sys.path.insert(0, EXTERNAL_CODES)

from Solver import Solver
from inference import compute_iou, compute_dice
from tqdm import tqdm

# Reuse the existing OmniCrackDataset from `crack_seg.py` to keep dataset behaviour consistent
sys.path.insert(0, os.path.dirname(__file__))
from crack_seg import OmniCrackDataset as CrackSegOmniDataset
from crack_seg import hybrid_loss
import matplotlib.pyplot as plt
import shutil


class ExternalWrapperDataset(Dataset):
    """Wrap the existing `CrackSegOmniDataset` so each item returns (image, GT, filename)
    which is what the external `Solver` expects from its data loader.
    """
    def __init__(self, root_dir, split, img_size=256):
        self.inner = CrackSegOmniDataset(root_dir, split=split, img_size=img_size)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, mask = self.inner[idx]
        # inner dataset uses glob and returns image path order; recover filename
        # The inner dataset stores paths in `img_paths` attribute
        try:
            img_path = self.inner.img_paths[idx]
            fname = os.path.basename(img_path)
        except Exception:
            fname = f"img_{idx}.png"

        return img, mask, fname


def make_loaders(root, img_size, batch_size, num_workers):
    train_ds = ExternalWrapperDataset(root, split='training', img_size=img_size)
    val_ds = ExternalWrapperDataset(root, split='validation', img_size=img_size)
    test_ds = ExternalWrapperDataset(root, split='test', img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader


def main(args):
    # configure
    cfg = SimpleNamespace()
    cfg.img_ch = 3
    cfg.output_ch = 1
    cfg.image_height = args.img_size
    cfg.image_width = args.img_size
    cfg.num_workers = args.num_workers
    cfg.lr = args.lr
    cfg.num_epochs = args.epochs
    cfg.num_epochs_decay = args.num_epochs_decay
    cfg.batch_size = args.batch_size
    cfg.loss_threshold = 0.5
    cfg.beta1 = 0.9
    cfg.beta2 = 0.999
    cfg.loss_weight = 2.61
    cfg.augmentation_prob = 0.15
    cfg.mode = 'train'
    cfg.report_name = args.report_name
    cfg.dataset = 'OmniCrack30k'
    cfg.model_type = 'EMT'
    cfg.model_path = args.model_path
    cfg.result_path = args.result_path
    cfg.SR_path = args.SR_path
    cfg.max_steps = args.max_steps
    cfg.train_path = ''
    cfg.valid_path = ''
    cfg.test_path = ''

    # create loaders
    train_loader, val_loader, test_loader = make_loaders(args.dataset_root, args.img_size, args.batch_size, args.num_workers)

    # build solver and run training
    solver = Solver(cfg, train_loader, val_loader, test_loader)
    solver.train()

    # After training, evaluate with our IoU and Dice
    net_path = os.path.join(cfg.model_path, cfg.report_name + '.pkl')
    net_path_state = net_path.replace('.pkl', '.pth')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # prefer using the in-memory model from Solver if available (avoids pickle issues)
    model = None
    if hasattr(solver, 'model') and solver.model is not None:
        model = solver.model
        print('Using in-memory trained model for evaluation')
        # also save a state_dict for future loads
        try:
            torch.save(model.state_dict(), net_path_state)
            print(f'Saved model state_dict to {net_path_state}')
        except Exception:
            pass
    else:
        # try to load from checkpoint file
        if not os.path.isfile(net_path):
            print('Model checkpoint not found at', net_path)
            return

        print('Loading trained model from', net_path)
        try:
            # try loading as whole object (may fail on newer PyTorch due to safe unpickling)
            model = torch.load(net_path, map_location=device)
        except Exception as e:
            print('torch.load failed:', e)
            # try loading a state_dict if present
            if os.path.isfile(net_path_state):
                print('Attempting to load state_dict from', net_path_state)
                # need to construct model architecture from external codes
                from EMT import EMT
                # the external EMT constructor used in Solver: EMT([[96,96], [96,128,128], [128,192,192,192], [128,128,128]], self.output_ch)
                arch = [[96,96], [96,128,128], [128,192,192,192], [128,128,128]]
                model = EMT(arch, cfg.output_ch).to(device)
                state = torch.load(net_path_state, map_location=device)
                model.load_state_dict(state)
            else:
                raise

    model.to(device)
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    n = 0
    with torch.no_grad():
        test_total = len(test_loader)
        if cfg.max_steps is not None:
            test_total = min(test_total, int(cfg.max_steps))

        step_test = 0
        for img, mask, name in tqdm(test_loader, total=test_total, desc='Test'):
            step_test += 1
            img = img.to(device)
            mask = mask.to(device)
            logits = model(img)
            # compute metrics using functions from inference.py
            iou = compute_iou(logits, mask)
            dice = compute_dice(logits, mask)
            total_iou += iou
            total_dice += dice
            n += 1

            if (cfg.max_steps is not None) and (step_test >= int(cfg.max_steps)):
                break

    if n > 0:
        print(f'Test IoU: {total_iou / n:.4f}, Test Dice: {total_dice / n:.4f}')
    else:
        print('No test samples found')

    # Locate external solver loss plot if it exists
    external_loss_plot = os.path.join(cfg.result_path, f"{cfg.report_name}_Loss_results.png")
    if os.path.isfile(external_loss_plot):
        print('External solver loss plot found at:', external_loss_plot)
        # copy to result_path root for quick access
        try:
            dst = os.path.join(cfg.result_path, os.path.basename(external_loss_plot))
            shutil.copy(external_loss_plot, dst)
        except Exception:
            pass
    else:
        # Fallback: compute train/val loss curves by running the model over loaders
        try:
            print('Computing train/val loss curves locally...')
            model.eval()
            train_losses = []
            val_losses = []

            with torch.no_grad():
                for imgs, masks in tqdm(train_loader, desc='Compute Train Loss'):
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    logits = model(imgs)
                    loss = hybrid_loss(logits, masks).item()
                    train_losses.append(loss)

                for imgs, masks in tqdm(val_loader, desc='Compute Val Loss'):
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    logits = model(imgs)
                    loss = hybrid_loss(logits, masks).item()
                    val_losses.append(loss)

            # plot
            plt.figure(figsize=(8,5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve ({cfg.report_name})')
            plt.legend()
            out_path = os.path.join(cfg.result_path, f'{cfg.report_name}_loss_curve_local.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print('Saved local loss curve to', out_path)
        except Exception as e:
            print('Failed to compute local loss curves:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, default='dataset/omnicrack30k', help='path to OmniCrack30k root')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum training steps per epoch (quick smoke test)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num-epochs-decay', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model-path', type=str, default='checkpoints/external_EMT/')
    parser.add_argument('--result-path', type=str, default='results/external_EMT/')
    parser.add_argument('--SR-path', type=str, default='results/external_EMT/SR/')
    parser.add_argument('--report-name', type=str, default='OmniCrack_EMT')

    args = parser.parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.SR_path, exist_ok=True)

    main(args)
