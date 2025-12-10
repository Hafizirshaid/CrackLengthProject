import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# 2D sinusoidal positional encoding
# --------------------------------------------------------
class SinusoidalPositionalEncoding2D(nn.Module):
    """
    2D sin/cos positional encoding.
    Given (B, C, H, W) returns x + pos of same shape.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin/cos."
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        dim_half = C // 2
        dim_quarter = dim_half // 2

        y = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)   # (H, 1)
        x_pos = torch.arange(W, device=device, dtype=dtype).unsqueeze(1)  # (1, W)

        omega_y = torch.arange(dim_quarter, device=device, dtype=dtype)
        omega_y = 1.0 / (10000 ** (2 * omega_y / dim_half))

        omega_x = torch.arange(dim_quarter, device=device, dtype=dtype)
        omega_x = 1.0 / (10000 ** (2 * omega_x / dim_half))

        y_emb = y * omega_y         # (H, dim_quarter)
        x_emb = x_pos * omega_x     # (W, dim_quarter)

        pos_y = torch.cat([torch.sin(y_emb), torch.cos(y_emb)], dim=1)  # (H, dim_half)
        pos_x = torch.cat([torch.sin(x_emb), torch.cos(x_emb)], dim=1)  # (W, dim_half)

        pos_y = pos_y.unsqueeze(1).expand(H, W, dim_half)  # (H, W, dim_half)
        pos_x = pos_x.unsqueeze(0).expand(H, W, dim_half)  # (H, W, dim_half)

        pos = torch.cat([pos_y, pos_x], dim=-1)   # (H, W, C)
        pos = pos.permute(2, 0, 1).unsqueeze(0)   # (1, C, H, W)
        return x + pos


# --------------------------------------------------------
# Simple ViT-style encoder
# --------------------------------------------------------
class SimpleViTEncoder(nn.Module):
    """
    Input: x in (B, 3, H, W)

    Steps:
      1) Conv patch embedding -> (B, C, H', W')
      2) Add 2D sinusoidal pos encoding
      3) Flatten -> Transformer -> seq (B, N, C)
      4) Reshape back to feat_grid (B, C, H', W')

    Returns:
      seq:       (B, N, C)
      feat_grid: (B, C, H', W')
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_encoding = SinusoidalPositionalEncoding2D(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,  # (B, N, C)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape

        # (B, C, H', W')
        feat_grid = self.patch_embed(x)
        feat_grid = self.pos_encoding(feat_grid)

        B, C, Hp, Wp = feat_grid.shape
        N = Hp * Wp

        # (B, N, C)
        seq = feat_grid.flatten(2).transpose(1, 2)
        seq = self.transformer(seq)

        # back to grid (B, C, H', W')
        feat_grid = seq.transpose(1, 2).reshape(B, C, Hp, Wp)
        return seq, feat_grid


# --------------------------------------------------------
# Full model: ViT + deconv + dot-product mixing
# --------------------------------------------------------
class CrackSegMixtureModel(nn.Module):
    """
    Implements your idea:

      ViT -> tokens (N x C) and low-res grid (C x H' x W')

      Deconv:
        feat_grid (B, C, H', W') -> pix_feat (B, C, H, W)

      Patch head:
        tokens (B, N, C) -> patch_logits_2c (B, N, 2)
                         -> patch_probs (B, N, 1) = P(crack per patch)

      Mixing:
        m_i[h,w] = <token_i, pix_feat[h,w]>   -> m: (B, N, H, W)
        final[h,w] = sum_i m_i[h,w] * p_i     -> (B, 1, H, W)
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.encoder = SimpleViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # deconv to full resolution (inverse of patch embedding)
        self.deconv = nn.ConvTranspose2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # simple patch classification head: 2 classes
        self.patch_head = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W)

        Returns:
          pixel_logits:    (B, 1, H, W)     # final per-pixel logits
          patch_logits_2c: (B, N, 2)        # per-patch 2-class logits
          patch_probs:     (B, N, 1)        # P(crack per patch)
        """
        B, _, H, W = x.shape

        # 1) Encoder
        seq, feat_grid = self.encoder(x)   # seq: (B, N, C), feat_grid: (B, C, H', W')
        B, N, C = seq.shape
        _, _, Hp, Wp = feat_grid.shape
        assert N == Hp * Wp, "N must equal H'*W'"

        # 2) Deconv to full resolution: (B, C, H, W)
        pix_feat = self.deconv(feat_grid)  # ideally (B, C, H, W)

        # # if shapes don't match perfectly, fix with interpolate
        # if pix_feat.shape[-2] != H or pix_feat.shape[-1] != W:
        #     pix_feat = F.interpolate(pix_feat, size=(H, W), mode="bilinear", align_corners=False)

        # 3) Patch classification head
        patch_logits_2c = self.patch_head(seq)         # (B, N, 2)
        patch_probs_2c = F.softmax(patch_logits_2c, dim=-1)  # (B, N, 2)
        patch_probs = patch_probs_2c[..., 1:2]         # (B, N, 1) -> prob of crack class

        # 4) Dot-product mixing
        # m[b, n, h, w] = <seq[b, n, :], pix_feat[b, :, h, w]>
        # seq:     (B, N, C)
        # pix_feat:(B, C, H, W)
        # -> m: (B, N, H, W)
        m = torch.einsum("bnc,bchw->bnhw", seq, pix_feat)

        # weight with patch probs: (B, N, 1, 1)
        p = patch_probs.view(B, N, 1, 1)  # (B, N, 1, 1)

        weighted = m * p                  # (B, N, H, W)
        # sum over N and keep channel dim -> (B, 1, H, W)
        pixel_logits = weighted.sum(dim=1, keepdim=True)

        return pixel_logits

