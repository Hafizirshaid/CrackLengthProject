# crack_len_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# segmentation backbones (same style as in crack_seg.py)
from models.unet import UNet
from models.vit_based import CrackSegMixtureModel as ViTBasedSegModel
from models.unet_resnet import get_unet_resnet50


class CrackSegLenModel(nn.Module):
    """
    Wraps a segmentation model and adds an MLP head to predict crack length
    from the segmentation logits (or probabilities).

    forward(x) returns:
        seg_logits: (B,1,H,W)
        crack_len_pred: (B,)
    """
    def __init__(self, seg_model, pooled_hw=16, hidden_dim=256, use_probs=True):
        super().__init__()
        self.seg_model = seg_model
        self.pooled_hw = pooled_hw
        self.use_probs = use_probs

        # assume seg_model outputs (B,1,H,W)
        in_dim = pooled_hw * pooled_hw  # 1 channel
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        seg_logits = self.seg_model(x)  # (B,1,H,W)

        if self.use_probs:
            feat_map = torch.sigmoid(seg_logits)
        else:
            feat_map = seg_logits

        # spatial pooling to fixed size
        pooled = F.adaptive_avg_pool2d(feat_map, (self.pooled_hw, self.pooled_hw))  # (B,1,p,p)
        feat = pooled.view(pooled.size(0), -1)  # (B, p*p)
        crack_len_pred = self.mlp(feat).squeeze(-1)  # (B,)

        return seg_logits, crack_len_pred


def build_seg_model(model_name: str, device: torch.device):
    """
    Build segmentation backbone similar to crack_seg.py.

    Only dense seg models are supported here.
    """
    model_name = model_name.lower()
    if model_name == "unet":
        seg_model = UNet().to(device)
    elif model_name == "vit_based":
        seg_model = ViTBasedSegModel().to(device)
    elif model_name == "unet_resnet":
        seg_model = get_unet_resnet50(
            num_classes=1,
            pretrained=True,
            freeze_encoder=False,
            bilinear=True,
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported seg model '{model_name}'. "
            f"Use one of: unet, vit_based, unet_resnet."
        )
    return seg_model


def create_crack_len_model(
    model_name: str,
    device: torch.device,
    seg_ckpt_path: str = "",
    pooled_hw: int = 16,
    hidden_dim: int = 256,
    use_probs: bool = True,
):
    """
    Convenience factory:
      - builds seg backbone
      - optionally loads seg-only checkpoint
      - wraps in CrackSegLenModel

    Returns:
        model: CrackSegLenModel on `device`
    """
    seg_model = build_seg_model(model_name, device)

    if seg_ckpt_path:
        ckpt = torch.load(seg_ckpt_path, map_location=device)
        # assume this is a state_dict for *seg_model* only
        seg_model.load_state_dict(ckpt)
        print(f"[create_crack_len_model] Loaded seg checkpoint from {seg_ckpt_path}")

    model = CrackSegLenModel(
        seg_model=seg_model,
        pooled_hw=pooled_hw,
        hidden_dim=hidden_dim,
        use_probs=use_probs,
    ).to(device)

    return model
