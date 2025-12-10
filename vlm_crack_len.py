import os
import csv
import re
import json
import numpy as np
import torch
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF

from transformers import (
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)

from dataset import KaggleCrackLenDataset
from train_eval_crack_len import make_loaders


# -----------------------------
# Utils
# -----------------------------
def parse_float_from_text(text: str) -> float:
    text = text.strip()
    if text == "":
        return float("nan")

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if m:
        try:
            val = float(m.group(0))
            # crack length can't be negative
            return max(val, 0.0)
        except ValueError:
            return float("nan")
    return float("nan")


def build_chat_texts(processor, prompts):
    """
    Try to build chat-style prompts if the processor supports it.
    If not (e.g., BLIP-2 / InstructBLIP), just return the raw prompts.
    """
    if not hasattr(processor, "apply_chat_template"):
        return prompts

    conversations = []
    for p in prompts:
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": p},
                    ],
                }
            ]
        )

    try:
        chat_texts = processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        return chat_texts
    except (ValueError, TypeError):
        # Processor has no usable chat template → fall back to plain prompts
        return prompts


# -----------------------------
# Inference
# -----------------------------
def infer_vlm_on_test(test_loader, model, processor, device="cuda", model_name=""):
    model.eval()
    preds, gts = [], []

    desc = f"{model_name} Test Inference" if model_name else "VLM Test Inference"

    with torch.no_grad():
        for imgs, masks, lengths in tqdm(test_loader, desc=desc):
            pil_imgs = [TF.to_pil_image(img) for img in imgs]

            base_prompt = (
                "You are measuring concrete cracks. "
                "Estimate the total crack length in this image in centimeters. "
                "Respond with ONLY a single numeric value such as 12.3. "
                "Do not output any words or units."
            )
            prompts = [base_prompt] * len(pil_imgs)

            texts_in = build_chat_texts(processor, prompts)

            inputs = processor(
                images=pil_imgs,
                text=texts_in,
                return_tensors="pt",
                padding=True,
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                num_beams=1,
            )
            texts = processor.batch_decode(outputs, skip_special_tokens=True)

            # Uncomment for debugging raw outputs
            # for raw in texts:
            #     print("[RAW VLM ANSWER]", repr(raw))

            for txt, gt in zip(texts, lengths):
                preds.append(parse_float_from_text(txt))
                gts.append(gt.item())

    preds = np.array(preds, dtype=float)
    gts = np.array(gts, dtype=float)

    mask = ~np.isnan(preds)
    n_drop = int(np.sum(~mask))
    if n_drop > 0:
        print(f"Dropped {n_drop} invalid predictions (NaNs).")

    preds = preds[mask]
    gts = gts[mask]

    if preds.size == 0:
        print("No valid predictions; cannot compute metrics.")
        return preds, gts

    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))

    tag = model_name if model_name else "VLM"
    print(f"[{tag}] Test MAE:  {mae:.4f} cm")
    print(f"[{tag}] Test RMSE: {rmse:.4f} cm")

    return preds, gts


# -----------------------------
# Model loader
# -----------------------------
def load_vlm(model_name: str, device: str = "cuda"):
    """
    Map short aliases -> real HF repo IDs and load with the right class.
    """
    name_lower = model_name.lower()

    # --- Alias mapping ---
    alias_map = {
        "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "deepseek-vl": "deepseek-ai/DeepSeek-VL2",
        "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
        "blip2": "Salesforce/blip2-flan-t5-xl",
        "blip2-flan-t5-xl": "Salesforce/blip2-flan-t5-xl",
        "instructblip": "Salesforce/instructblip-flan-t5-xl",
        "instructblip-flan-t5-xl": "Salesforce/instructblip-flan-t5-xl",
    }
    if model_name in alias_map:
        hf_id = alias_map[model_name]
        print(f"[Alias] Mapping '{model_name}' -> '{hf_id}'")
        model_name = hf_id
        name_lower = model_name.lower()

    # --- LLaVA family (AutoModelForImageTextToText) ---
    if "llava" in name_lower:
        print(f"[Model] Loading LLaVA-style model: {model_name}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    # --- Qwen2.5-VL family (Vision2Seq + trust_remote_code) ---
    if "qwen2.5-vl" in name_lower or "qwen2-vl" in name_lower:
        print(f"[Model] Loading Qwen-VL-style model: {model_name}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        return model, processor

    # --- DeepSeek-VL family (Vision2Seq + trust_remote_code) ---
    if "deepseek-vl" in name_lower or "deepseek-vl2" in name_lower:
        print(f"[Model] Loading DeepSeek-VL-style model: {model_name}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        return model, processor

    # --- BLIP-2 family (single-GPU) ---
    if "salesforce/blip2" in name_lower or name_lower.startswith("blip2"):
        print(f"[Model] Loading BLIP-2 model: {model_name}")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        model = model.to(device)
        processor = Blip2Processor.from_pretrained(model_name)
        return model, processor

    # --- InstructBLIP family (single-GPU) ---
    if "salesforce/instructblip" in name_lower or name_lower.startswith("instructblip"):
        print(f"[Model] Loading InstructBLIP model: {model_name}")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        model = model.to(device)
        processor = InstructBlipProcessor.from_pretrained(model_name)
        return model, processor

    # --- Generic fallback ---
    print(f"[Model] Loading generic VLM with AutoModelForImageTextToText: {model_name}")
    print("        (If this fails, add a specific branch in load_vlm.)")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    return model, processor


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help=(
            "HF model id or short alias. "
            "Examples: llava-hf/llava-1.5-7b-hf, qwen2.5-vl, "
            "Qwen/Qwen2.5-VL-7B-Instruct, deepseek-vl, blip2, instructblip"
        ),
    )
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    _, _, test_loader = make_loaders(
        root=args.root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print("Test batches:", len(test_loader))

    # Load chosen VLM (supports aliases)
    model, processor = load_vlm(args.model, device=device)
    # NOTE: don't call model.to(device) here; handled inside load_vlm.

    preds, gts = infer_vlm_on_test(
        test_loader=test_loader,
        model=model,
        processor=processor,
        device=device,
        model_name=args.model,
    )

    result_json = {"results": []}
    for i in range(len(gts)):
        result_json["results"].append(
            {
                "gt": float(gts[i]),
                "pred": float(preds[i]),
            }
        )

    result_json["rmse"] = float(np.sqrt(np.mean((preds - gts) ** 2)))
    result_json["mae"] = float(np.mean(np.abs(preds - gts)))

    if args.out_csv is not None and preds.size > 0:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "gt_cm", "pred_cm"])
            for i, (gt, pred) in enumerate(zip(gts, preds)):
                w.writerow([i, gt, pred])
        print("Saved CSV to:", args.out_csv)

    if args.out_json is not None and preds.size > 0:
        out_json_path = args.out_json
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(result_json, f, indent=4)
        print("Saved results JSON to:", out_json_path)
