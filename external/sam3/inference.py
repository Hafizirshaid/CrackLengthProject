# ...existing code...
import torch
#################################### For Image ####################################
from PIL import Image, ImageDraw, ImageFont
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
image_path = "../../dataset/omnicrack30k/images/training/BCL_c1.png"
image = Image.open(image_path).convert("RGB")
inference_state = processor.set_image(image)

# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="crack")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# ...existing code...
def visualize_segmentation(image_pil, masks, boxes=None, scores=None, save_path=None, max_masks=10, alpha=0.5):
    """
    Overlay binary masks, draw boxes and scores on the image and display/save result.

    - image_pil: PIL.Image RGB
    - masks: torch.Tensor or np.ndarray of shape (N, H, W) or list of masks
    - boxes: torch.Tensor or np.ndarray of shape (N,4) in (x1,y1,x2,y2)
    - scores: iterable of length N
    - save_path: path to save visualization
    - max_masks: limit number of masks to overlay
    - alpha: blending alpha for mask overlay (0..1)
    """
    # Convert masks to numpy arrays
    if hasattr(masks, "detach"):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.array(masks)

    # Normalize common unexpected shapes to (N, H, W)
    # Cases handled: (H,W) -> (1,H,W), (H,W,C) -> (C,H,W), (N,H,W,1) -> (N,H,W), (N,1,H,W) -> (N,H,W)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    elif masks_np.ndim == 3:
        # could be (N,H,W) or (H,W,C)
        h, w = image_pil.height, image_pil.width
        a, b, c = masks_np.shape
        if a == h and b == w:
            # (H, W, C) -> (C, H, W)
            masks_np = np.transpose(masks_np, (2, 0, 1))
        else:
            # assume (N, H, W)
            pass
    elif masks_np.ndim == 4:
        # common shapes: (N, H, W, 1) or (N, 1, H, W)
        if masks_np.shape[-1] == 1:
            masks_np = masks_np[..., 0]
        elif masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0, ...]
        else:
            # try squeeze fallback
            masks_np = np.squeeze(masks_np)
            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]

    # ensure at least (N, H, W)
    if masks_np.ndim != 3:
        raise ValueError(f"Unsupported mask array shape after normalization: {masks_np.shape}")

    # Convert masks to boolean/0..1 floats
    # If mask values are in {0,255}, scale to 0..1
    if masks_np.dtype == np.uint8 or masks_np.max() > 1:
        masks_np = (masks_np.astype(np.float32) / 255.0)
    masks_np = (masks_np > 0.5).astype(np.uint8)

    N = masks_np.shape[0]
    Nvis = min(N, max_masks)

    base = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))

    # deterministic colors
    rng = random.Random(0)
    colors = [tuple(rng.randint(50, 255) for _ in range(3)) for _ in range(Nvis)]

    for i in range(Nvis):
        mask = masks_np[i]
        mask_h, mask_w = mask.shape
        if (mask_w, mask_h) != (base.width, base.height):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((base.width, base.height), resample=Image.NEAREST)
        else:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        color = colors[i]
        color_img = Image.new("RGBA", base.size, color + (0,))
        mask_alpha = mask_img.point(lambda p: int(p * alpha))
        color_img.putalpha(mask_alpha)
        overlay = Image.alpha_composite(overlay, color_img)

    composed = Image.alpha_composite(base, overlay).convert("RGB")

    # draw boxes and scores with PIL
    draw = ImageDraw.Draw(composed)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if boxes is not None:
        if hasattr(boxes, "detach"):
            boxes_np = boxes.detach().cpu().numpy()
        else:
            boxes_np = np.array(boxes)
        for i in range(min(N, Nvis)):
            x1, y1, x2, y2 = boxes_np[i]
            draw.rectangle([x1, y1, x2, y2], outline=colors[i], width=2)
            if scores is not None:
                sc = scores[i]
                if hasattr(sc, "detach"):
                    try:
                        sc = float(sc.detach().cpu().item())
                    except Exception:
                        sc = float(sc)
                else:
                    sc = float(sc)
                text = f"{sc:.2f}"
                text_pos = (x1 + 3, max(0, y1 - 10))
                draw.text(text_pos, text, fill=colors[i], font=font)

    # show inline (if running interactively) and save
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(composed)
    if save_path is None:
        save_dir = os.path.dirname(image_path)
        save_path = os.path.join(save_dir, "sam3_inference_vis.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to {save_path}")
    return composed
# ...existing code...

# Call visualization
_visualization_path = 'viz_sam/sam3_inference_visualization.png'
os.makedirs(os.path.dirname(_visualization_path), exist_ok=True)
visualize_segmentation(image, masks, boxes=boxes, scores=scores, save_path=_visualization_path, max_masks=10, alpha=0.45)
# ...existing code...