import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import re
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF
from PIL import Image

from transformers import AutoModelForCausalLM

# Requires:
#   git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
#   cd DeepSeek-VL2
#   pip install -e .
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from train_eval_crack_len import make_loaders


# -------------------------
# Helpers
# -------------------------
def parse_float_from_text(text: str) -> float:
    """
    Extract the first floating-point number from text, or NaN if none.
    """
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return float("nan")
    return float("nan")


def build_deepseek_conversation(prompt: str):
    """
    Build a single-image DeepSeek-VL2 conversation.

    Format follows the official README:
    [
        {"role": "<|User|>", "content": "<image>\\n ..."},
        {"role": "<|Assistant|>", "content": ""},
    ]
    """
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n" + prompt,
        },
        {
            "role": "<|Assistant|>",
            "content": "",
        },
    ]
    return conversation


# -------------------------
# DeepSeek-VL2 loading
# -------------------------
def load_deepseek(model_name: str, device: str = "cuda"):
    """
    Load DeepSeek-VL2 model and processor.

    We explicitly import DeepseekVLV2Processor / DeepseekVLV2ForCausalLM
    so that their config/model types are registered with Transformers.
    """
    print(f"[Model] Loading DeepSeek-VL2 model: {model_name}")

    # Processor (handles conversations + images)
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
        model_name
    )
    tokenizer = processor.tokenizer

    # Model (vision-language causal LM)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device).eval()

    return model, processor, tokenizer


# -------------------------
# Inference loop
# -------------------------
def infer_deepseek_on_test(
    test_loader,
    model,
    processor,
    tokenizer,
    device: str = "cuda",
    model_name: str = "",
):
    model.eval()
    torch.set_grad_enabled(False)

    preds, gts = [], []

    desc = f"{model_name} Test Inference" if model_name else "DeepSeek-VL2 Test Inference"

    base_prompt = (
        "You are measuring concrete cracks. "
        "Estimate the total crack length in this image in centimeters. "
        "Answer with only a single number (no units, no extra text)."
    )

    for imgs, masks, lengths in tqdm(test_loader, desc=desc):
        # imgs: (B, C, H, W) tensor in [0,1] or [0,255]
        pil_imgs = [
            TF.to_pil_image(img) if not isinstance(img, Image.Image) else img
            for img in imgs
        ]

        for pil_img, gt_len in zip(pil_imgs, lengths):
            conversation = build_deepseek_conversation(base_prompt)

            # Prepare inputs for DeepSeek-VL2
            prepare_inputs = processor(
                conversations=conversation,
                images=[pil_img],
                force_batchify=True,
                system_prompt="",
            ).to(device)

            # Run image encoder to get the embeddings
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

            # IMPORTANT: use `model.language.generate`, not `language_model`
            output_ids = model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=32,
                do_sample=False,
                use_cache=True,
            )

            text = tokenizer.decode(
                output_ids[0].cpu().tolist(),
                skip_special_tokens=True,
            )

            pred_val = parse_float_from_text(text)
            preds.append(pred_val)
            # gt_len might be a tensor; convert to float
            gts.append(float(gt_len))

    preds = np.array(preds, dtype=float)
    gts = np.array(gts, dtype=float)

    # Drop NaNs
    mask_valid = ~np.isnan(preds)
    n_drop = int((~mask_valid).sum())
    if n_drop > 0:
        print(f"Dropped {n_drop} invalid predictions (NaNs).")

    preds = preds[mask_valid]
    gts = gts[mask_valid]

    if preds.size == 0:
        print("No valid predictions; cannot compute metrics.")
        return preds, gts

    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))

    tag = model_name if model_name else "DeepSeek-VL2"
    print(f"[{tag}] Test MAE:  {mae:.4f} cm")
    print(f"[{tag}] Test RMSE: {rmse:.4f} cm")

    return preds, gts


# -------------------------
# Main
# -------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-vl2-tiny",
        help="HF model id for DeepSeek-VL2 (e.g., deepseek-ai/deepseek-vl2-tiny).",
    )
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # Reuse your existing loaders
    _, _, test_loader = make_loaders(
        root=args.root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print("Test batches:", len(test_loader))

    model, processor, tokenizer = load_deepseek(args.model, device=device)

    preds, gts = infer_deepseek_on_test(
        test_loader=test_loader,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        model_name=args.model,
    )

    # Build JSON summary
    result_json = {"results": []}
    for gt, pred in zip(gts, preds):
        result_json["results"].append(
            {
                "gt": float(gt),
                "pred": float(pred),
            }
        )

    if preds.size > 0:
        result_json["rmse"] = float(np.sqrt(np.mean((preds - gts) ** 2)))
        result_json["mae"] = float(np.mean(np.abs(preds - gts)))

    # Save CSV
    if args.out_csv is not None and preds.size > 0:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "gt_cm", "pred_cm"])
            for i, (gt, pred) in enumerate(zip(gts, preds)):
                w.writerow([i, gt, pred])
        print("Saved CSV to:", args.out_csv)

    # Save JSON
    if args.out_json is not None and preds.size > 0:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result_json, f, indent=4)
        print("Saved results JSON to:", args.out_json)


if __name__ == "__main__":
    main()
