import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from skimage.morphology import disk
from sklearn.metrics import jaccard_score
import torch
from models.unet_resnet import get_unet_resnet50
from torchvision.transforms import Normalize
from PIL import Image
import torchvision.transforms as T
from skimage.morphology import thin
from models.vit_based import CrackSegMixtureModel as ViTBasedSegModel
from models.unet import UNet

# from omnicrack30k.inference import OmniCrack30kModel

SUBSETS = {'validation':
               ['BCL', 'Ceramic', 'CFD', 'CRACK500', 'CrackTree260', 'CrSpEE', 'CSSC', 'DeepCrack',
                'DIC', 'GAPS384', 'Khanh11k', 'LCW', 'Masonry', 'S2DS', 'TopoDS', 'UAV75'],
           'test':
               ['AEL', 'BCL', 'Ceramic', 'CFD', 'CRACK500', 'CrackLS315', 'CRKWH100', 'CrSpEE', 'CSSC',
                'DeepCrack', 'DIC', 'GAPS384', 'Khanh11k', 'LCW', 'Masonry', 'S2DS', 'Stone331', 'TopoDS',
                'UAV75']}


def apply_tolerance(true, pred, tol=5):
    true_dil = cv2.dilate(true, disk(tol), iterations=1)
    pred_dil = cv2.dilate(pred, disk(tol), iterations=1)

    # infer true/false positives and negatives
    tp = true * pred_dil
    fp = pred - (pred * true_dil)
    fn = true - tp

    true, pred = tp + fn, tp + fp
    return true, pred


def run_evaluation(datapath, texpath, checkpoints, model, split, subset=None, tolerance=4, planpath=None, folds=(0, 1, 2, 3, 4)):
    # predictor = OmniCrack30kModel(planpath=planpath, folds=folds, allow_tqdm=False)
    print(f"Evaluating using {model} model...")
    print(f"Loading model checkpoint from: {checkpoints}")
    print(f"Using data from: {datapath}")
    print("split:", split)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model == 'unet':
        predictor = UNet().to(device)
    elif model == 'vit_based':
        predictor = ViTBasedSegModel().to(device)
    elif model == "unet_resnet":
        predictor = get_unet_resnet50(num_classes=1,
                                  pretrained=True,
                                  freeze_encoder=False,
                                  bilinear=True).to(device)
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    predictor.load_state_dict(torch.load(checkpoints, map_location=device))
    predictor.eval()

    subsets = [subset] if subset is not None else SUBSETS[split]

    trues = {key: np.empty((0,), dtype=bool) for key in subsets}
    preds = {key: np.empty((0,), dtype=bool) for key in subsets}

    tex = ""
    classes = ["background", "crack"]
    for key in subsets:
        img_paths = (datapath / "images" / split).glob(f"{key}*.png")

        for f in tqdm(list(img_paths)):
            # load image and annotation
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            true = cv2.imread(str((datapath / "centerlines" / split / f.name)), cv2.IMREAD_GRAYSCALE)

            # run inference and map classes

            # prepare image
            # img = torch.tensor(img, dtype=torch.float32)
            # img = img.moveaxis(-1, 0) if img.shape[0] != 3 else img
            # img[:, torch.all(img == 0, dim=0)] = \
                # torch.rand_like(img)[:, torch.all(img == 0, dim=0)]  # avoid patches of all zeros
            rgb = False
            # transform image
            # img = img[[2, 1, 0], ...] if rgb else img
            # data = Normalize(img.mean((1, 2)), img.std((1, 2)))(img)
            # data = data.unsqueeze(1)
            img_size = 256

            img_tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
            ])

            img = img_tf(Image.fromarray(img.astype(np.uint8))).to(device)
            logits = predictor(img.unsqueeze(0)).squeeze().cpu()

            # For binary segmentation (single-channel logits) use sigmoid to get probabilities
            logits = logits.to(torch.float32)
            # Handle both (1, H, W) and (H, W) cases robustly
            if logits.dim() == 3 and logits.shape[0] == 1:
                prob = torch.sigmoid(logits[0])
            else:
                prob = torch.sigmoid(logits)
            prob_np = prob.detach().cpu().numpy()

            # threshold to binary prediction (0 background, 1 crack)
            pred_bin = (prob_np > 0.5).astype(np.uint8)
            # visualization map 0-255
            argmax = (pred_bin * 255).astype(np.uint8)

            # compute centerlines (thin expects a 2D binary array)
            centerlines = np.uint8(255 * thin(pred_bin))

            # invert for visualization
            softmax = 1 - prob_np
            argmax = 255 - argmax
            centerlines = 255 - centerlines

            true, pred = np.uint8(true == 0), np.uint8(centerlines == 0)

            # some subsets (e.g. Stone331) provide resized masks -> correct that
            if pred.shape != true.shape:
                pred = cv2.resize(pred, (true.shape[1], true.shape[0]), interpolation=cv2.INTER_NEAREST)

            # apply tolerance
            true, pred = apply_tolerance(true, pred, tol=tolerance)
            true, pred = true.flatten(), pred.flatten()

            # remove true negatives (they do not affect IoU)
            keep_idxs = np.where((true == 1) + (pred == 1))[0]
            true, pred = true[keep_idxs], pred[keep_idxs]

            # store results
            trues[key] = np.append(trues[key], true)
            preds[key] = np.append(preds[key], pred)

        # compute centerline IoU (clIoU)
        cliou = jaccard_score(trues[key], preds[key])
        tex += f"& {100 * cliou:.1f} "
        print(f"\n{key}\t{cliou:.3f}")

    if texpath is not None:
        with open(texpath, 'w') as f:
            f.write(tex)
        print(f"Wrote LaTeX results to: {texpath}")


if __name__ == "__main__":
    parser = ArgumentParser(description="""Run evaluation and compute centerline IoU (clIoU).""")
    parser.add_argument('split', default="test", nargs='?', choices=['test', 'validation'], help="Split for evaluation")
    parser.add_argument('-s', '--subset', type=str, default=None, help="Subset to evaluate.")
    parser.add_argument('-t', '--tolerance', type=int, default=4, help="Tolerance of the clIoU.")
    parser.add_argument('-p', '--planpath', type=str, default=None, help="Path to the plan, i.e. model and weights.")
    parser.add_argument('-f', '--folds', nargs="+", type=int, default=(0, 1, 2, 3, 4), help="Folds for ensemble.")
    # parser.add_argument('-tp', '--texpath', type=str, default='results.tex', help="Path to latex output file, if desired.")
    parser.add_argument('--datapath', default="/mnt/home/irshaid2/crack_seg/omnicrack30k", help="Path to root folder of omnicrack30k dataset.")
    parser.add_argument('--model', type=str, default="unet_resnet", choices=['unet', 'vit_based', 'unet_resnet'], help="Model type to evaluate.")
    parser.add_argument('-c', '--checkpoints', type=str, default="checkpoints/unet_resnet_lr_0.0001_bs_32_best.pth", help="Path to the trained model checkpoint.")
    args = parser.parse_args()
    textpath = f"results_vit_based_lr_0.0001_bs_8_best_{args.split}.tex"

    run_evaluation(Path(args.datapath), textpath, args.checkpoints, args.model, args.split, args.subset, args.tolerance, args.planpath, args.folds)
    