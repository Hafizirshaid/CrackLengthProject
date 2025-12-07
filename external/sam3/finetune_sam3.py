import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:  # pragma: no cover - installation issue
    raise ImportError(
        "pycocotools is required to convert OmniCrack masks into COCO JSON format."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = PROJECT_ROOT / "external" / "sam3"
SAM3_TRAIN_DIR = SAM3_ROOT / "sam3" / "train"
SAM3_CONFIG_DIR = SAM3_TRAIN_DIR / "configs"
BPE_RELATIVE_PATH = Path("assets/bpe_simple_vocab_16e6.txt.gz")

SPLIT_MAP = {
    "train": "training",
    "val": "validation",
    "test": "test",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class CocoStats:
    images: int
    annotations: int


def ensure_exists(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return (np.array(mask) < 128).astype(np.uint8)


def build_coco_for_split(
    dataset_root: Path,
    split_key: str,
    output_json: Path,
    limit: Optional[int],
) -> CocoStats:
    image_dir = dataset_root / "images" / SPLIT_MAP[split_key]
    mask_dir = dataset_root / "annotations" / SPLIT_MAP[split_key]

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {mask_dir}")

    images: List[dict] = []
    annotations: List[dict] = []
    ann_id = 1

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if limit is not None:
        image_files = image_files[:limit]

    image_id = 0
    for img_path in tqdm(image_files, desc=f"COCO ({split_key})", unit="img"):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path.name}")

        mask = load_binary_mask(mask_path)
        if mask.sum() == 0:
            continue  # skip empty annotations completely

        pil_image = Image.open(img_path).convert("RGB")
        width, height = pil_image.size
        image_id += 1
        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )
        ys, xs = np.where(mask > 0)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        bbox = [
            float(x1),
            float(y1),
            float(x2 - x1 + 1),
            float(y2 - y1 + 1),
        ]
        area = float(mask.sum())
        segmentation = [
            [
                float(x1),
                float(y1),
                float(x2 + 1),
                float(y1),
                float(x2 + 1),
                float(y2 + 1),
                float(x1),
                float(y2 + 1),
            ]
        ]

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": area,
                "bbox": [float(x) for x in bbox],
                "iscrowd": 0,
                "noun_phrase": "crack",
            }
        )
        ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "crack",
                "supercategory": "damage",
            }
        ],
    }

    ensure_exists(output_json.parent)
    with output_json.open("w") as f:
        json.dump(coco_dict, f)

    return CocoStats(images=len(images), annotations=len(annotations))


def create_coco_annotations(
    dataset_root: Path,
    coco_root: Path,
    force: bool,
    limits: Dict[str, Optional[int]],
) -> Dict[str, Path]:
    ensure_exists(coco_root)
    split_to_json: Dict[str, Path] = {}
    for split_key in SPLIT_MAP.keys():
        json_path = coco_root / f"{split_key}.json"
        split_to_json[split_key] = json_path
        if json_path.exists() and not force:
            continue
        stats = build_coco_for_split(
            dataset_root, split_key, json_path, limit=limits.get(split_key)
        )
        print(
            f"[COCO:{split_key}] images={stats.images} annotations={stats.annotations} -> {json_path}"
        )
    return split_to_json


def render_config_text(
    dataset_root: Path,
    output_dir: Path,
    bpe_path: Path,
    coco_paths: Dict[str, Path],
    img_size: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    epochs: int,
    lr_scale: float,
    seed: int,
    run_name: str,
) -> str:
    train_images = (dataset_root / "images" / SPLIT_MAP["train"]).as_posix()
    val_images = (dataset_root / "images" / SPLIT_MAP["val"]).as_posix()
    test_images = (dataset_root / "images" / SPLIT_MAP["test"]).as_posix()

    train_json = coco_paths["train"].as_posix()
    val_json = coco_paths["val"].as_posix()
    test_json = coco_paths["test"].as_posix()

    experiment_dir = output_dir.as_posix()
    bpe_abs = bpe_path.as_posix()

    return f"""# @package _global_
defaults:
  - _self_

paths:
  dataset_root: {dataset_root.as_posix()}
  experiment_log_dir: {experiment_dir}
  bpe_path: {bpe_abs}

omnicrack:
  train_images: {train_images}
  val_images: {val_images}
  test_images: {test_images}
  train_json: {train_json}
  val_json: {val_json}
  test_json: {test_json}
  loss:
    _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
    matcher: ${{scratch.matcher}}
    o2m_weight: 2.0
    o2m_matcher:
      _target_: sam3.train.matcher.BinaryOneToManyMatcher
      alpha: 0.3
      threshold: 0.4
      topk: 4
    use_o2m_matcher_on_o2m_aux: false
    loss_fns_find:
      - _target_: sam3.train.loss.loss_fns.Boxes
        weight_dict:
          loss_bbox: 5.0
          loss_giou: 2.0
      - _target_: sam3.train.loss.loss_fns.IABCEMdetr
        weak_loss: False
        weight_dict:
          loss_ce: 20.0
          presence_loss: 20.0
        pos_weight: 10.0
        alpha: 0.25
        gamma: 2
        use_presence: True
        pos_focal: false
        pad_n_queries: 200
        pad_scale_pos: 1.0
      - _target_: sam3.train.loss.loss_fns.Masks
        focal_alpha: 0.25
        focal_gamma: 2.0
        weight_dict:
          loss_mask: 200.0
          loss_dice: 10.0
        compute_aux: false
    loss_fn_semantic_seg: null
    scale_by_find_batch_size: ${{scratch.scale_by_find_batch_size}}
  train_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes:
            _target_: sam3.train.transforms.basic.get_random_resize_scales
            size: ${{scratch.resolution}}
            min_size: 480
            rounded: false
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${{scratch.resolution}}
          square: true
          consistent_transform: ${{scratch.consistent_transform}}
        - _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
          size: ${{scratch.resolution}}
          consistent_transform: ${{scratch.consistent_transform}}
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${{scratch.train_norm_mean}}
          std: ${{scratch.train_norm_std}}
    - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
      query_filter:
        _target_: sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut
        max_num_objects: ${{scratch.max_ann_per_img}}
  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes: ${{scratch.resolution}}
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${{scratch.resolution}}
          square: true
          consistent_transform: False
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${{scratch.val_norm_mean}}
          std: ${{scratch.val_norm_std}}

scratch:
  enable_segmentation: true
  resolution: {img_size}
  consistent_transform: False
  max_ann_per_img: 200
  hybrid_repeats: 1
  train_norm_mean: [0.5, 0.5, 0.5]
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]
  train_batch_size: {batch_size}
  val_batch_size: {val_batch_size}
  num_train_workers: {num_workers}
  num_val_workers: {num_workers}
  max_data_epochs: {epochs}
  train_transforms: ${{omnicrack.train_transforms}}
  val_transforms: ${{omnicrack.val_transforms}}
  matcher:
    _target_: sam3.train.matcher.BinaryHungarianMatcherV2
    focal: true
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2
    stable: False
  scale_by_find_batch_size: True
  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${{scratch.hybrid_repeats}}
    dict_key: omnicrack
    with_seg_masks: ${{scratch.enable_segmentation}}
  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${{scratch.hybrid_repeats}}
    dict_key: omnicrack_val
    with_seg_masks: ${{scratch.enable_segmentation}}
  lr_scale: {lr_scale}
  lr_transformer: ${{times:8e-4,${{scratch.lr_scale}}}}
  lr_vision_backbone: ${{times:2.5e-4,${{scratch.lr_scale}}}}
  lr_language_backbone: ${{times:5e-5,${{scratch.lr_scale}}}}
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 20
  scheduler_cooldown: 20
  gather_pred_via_filesys: false

trainer:
  _target_: sam3.train.trainer.Trainer
  max_epochs: ${{scratch.max_data_epochs}}
  accelerator: cuda
  seed_value: {seed}
  val_epoch_freq: 1
  mode: train
  skip_first_val: False
  skip_saving_ckpts: False
  empty_gpu_mem_cache_after_eval: True
  distributed:
    backend: nccl
    find_unused_parameters: True
    gradient_as_bucket_view: True
  loss:
    all: ${{omnicrack.loss}}
    default:
      _target_: sam3.train.loss.sam3_loss.DummyLoss
  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        load_segmentation: ${{scratch.enable_segmentation}}
        img_folder: ${{omnicrack.train_images}}
        ann_file: ${{omnicrack.train_json}}
        transforms: ${{scratch.train_transforms}}
        max_ann_per_img: ${{scratch.max_ann_per_img}}
        multiplier: 1
        training: true
        use_caching: False
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: false
          category_chunk_size: 1
          _partial_: true
      shuffle: True
      batch_size: ${{scratch.train_batch_size}}
      num_workers: ${{scratch.num_train_workers}}
      pin_memory: True
      drop_last: True
      collate_fn: ${{scratch.collate_fn}}
    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        load_segmentation: ${{scratch.enable_segmentation}}
        img_folder: ${{omnicrack.val_images}}
        ann_file: ${{omnicrack.val_json}}
        transforms: ${{scratch.val_transforms}}
        max_ann_per_img: ${{scratch.max_ann_per_img}}
        multiplier: 1
        training: false
        use_caching: False
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: false
          category_chunk_size: 1
          _partial_: true
      shuffle: False
      batch_size: ${{scratch.val_batch_size}}
      num_workers: ${{scratch.num_val_workers}}
      pin_memory: True
      drop_last: False
      collate_fn: ${{scratch.collate_fn_val}}
  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${{paths.bpe_path}}
    device: cuda
    eval_mode: false
    enable_segmentation: ${{scratch.enable_segmentation}}
  meters:
    val:
      omnicrack_val: null
  optim:
    amp:
      enabled: True
      amp_dtype: bfloat16
    optimizer:
      _target_: torch.optim.AdamW
    gradient_clip:
      _target_: sam3.train.optim.optimizer.GradientClipper
      max_norm: 0.1
      norm_type: 2
    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: True
        layer_decay_value: ${{scratch.lrd_vision_backbone}}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides:
          - pattern: '*pos_embed*'
            value: 1.0
    options:
      lr:
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_transformer}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_vision_backbone}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
          param_names:
            - 'backbone.vision_backbone.*'
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_language_backbone}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
          param_names:
            - 'backbone.language_backbone.*'
      weight_decay:
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: ${{scratch.wd}}
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.0
          param_names:
            - '*bias*'
          module_cls_names: ['torch.nn.LayerNorm']
  checkpoint:
    save_dir: ${{launcher.experiment_log_dir}}/checkpoints
    save_freq: 1
  logging:
    tensorboard_writer:
      _target_: sam3.train.utils.logger.make_tensorboard_logger
      log_dir: ${{launcher.experiment_log_dir}}/tensorboard
      flush_secs: 120
      should_log: True
    wandb_writer: null
    log_dir: ${{launcher.experiment_log_dir}}/logs/{run_name}
    log_freq: 10

launcher:
  num_nodes: 1
  gpus_per_node: 1
  experiment_log_dir: ${{paths.experiment_log_dir}}
  multiprocessing_context: spawn

submitit:
  account: null
  partition: null
  qos: null
  timeout_hour: 48
  use_cluster: False
  cpus_per_task: {max(num_workers, 4)}
  port_range: [10000, 65000]
  constraint: null
  job_array:
    num_tasks: 1
    task_index: 0
"""


def write_config_file(config_path: Path, config_text: str) -> str:
    ensure_exists(config_path.parent)
    config_path.write_text(config_text)
    rel = config_path.relative_to(SAM3_TRAIN_DIR).as_posix()
    print(f"Wrote Hydra config -> {config_path}")
    return rel


def run_subprocess(cmd: List[str], workdir: Path) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=workdir, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def launch_training(config_rel_path: str, num_gpus: int, num_nodes: int) -> None:
    train_script = SAM3_TRAIN_DIR / "train.py"
    cmd = [
        sys.executable,
        str(train_script),
        "-c",
        config_rel_path,
        "--use-cluster",
        "0",
        "--num-gpus",
        str(num_gpus),
        "--num-nodes",
        str(num_nodes),
    ]
    run_subprocess(cmd, SAM3_ROOT)


def run_inference(
    checkpoint_path: Path,
    dataset_root: Path,
    img_size: int,
    score_threshold: float,
    splits: List[str],
    output_dir: Path,
    max_images: Optional[int],
) -> None:
    inference_script = PROJECT_ROOT / "external" / "sam3" / "inference_sam3.py"
    save_masks_dir = output_dir / "inference" / "masks"
    viz_dir = output_dir / "inference" / "viz"
    ensure_exists(save_masks_dir)
    ensure_exists(viz_dir)
    cmd = [
        sys.executable,
        str(inference_script),
        "--dataset_root",
        dataset_root.as_posix(),
        "--splits",
        *splits,
        "--checkpoint_path",
        checkpoint_path.as_posix(),
        "--img_size",
        str(img_size),
        "--score_threshold",
        str(score_threshold),
        "--save_masks_dir",
        save_masks_dir.as_posix(),
        "--viz_dir",
        viz_dir.as_posix(),
        "--viz_max_images",
        "16",
    ]
    if max_images is not None:
        cmd += ["--max_images", str(max_images)]
    run_subprocess(cmd, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper that prepares Hydra configs and runs SAM3 train.py."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/omnicrack30k",
        help="Dataset root with images/ and annotations/ folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/sam3_omnicrack",
        help="Directory for Hydra experiment logs/checkpoints.",
    )
    parser.add_argument(
        "--coco_dir",
        type=str,
        default=None,
        help="Optional directory to store generated COCO JSON files. Defaults to dataset_root/coco_annotations.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="omnicrack_finetune",
        help="Name used for config/log folders.",
    )
    parser.add_argument("--img_size", type=int, default=1008, help="Resize dimension.")
    parser.add_argument("--batch_size", type=int, default=2, help="Train batch size.")
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Validation batch size."
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument(
        "--lr_scale",
        type=float,
        default=0.1,
        help="Scales the base learning rates defined in the template.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="GPUs per node for train.py."
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Number of nodes for train.py."
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Score threshold used during inference mask fusion.",
    )
    parser.add_argument(
        "--inference_splits",
        nargs="+",
        default=["test"],
        choices=list(SPLIT_MAP.keys()),
        help="Splits evaluated after training.",
    )
    parser.add_argument(
        "--max_infer_images",
        type=int,
        default=None,
        help="Limit inference images per split (debug).",
    )
    parser.add_argument(
        "--limit_train_images",
        type=int,
        default=None,
        help="Optional cap on training images when building COCO JSON.",
    )
    parser.add_argument(
        "--limit_val_images",
        type=int,
        default=None,
        help="Optional cap on validation images when building COCO JSON.",
    )
    parser.add_argument(
        "--limit_test_images",
        type=int,
        default=None,
        help="Optional cap on test images when building COCO JSON.",
    )
    parser.add_argument(
        "--force_coco",
        action="store_true",
        help="Regenerate COCO annotations even if files already exist.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip calling train.py (useful if checkpoints already exist).",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip running inference after training.",
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="1",
        help="Which CUDA device index to expose (sets CUDA_VISIBLE_DEVICES).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    coco_dir = (
        Path(args.coco_dir).resolve()
        if args.coco_dir is not None
        else dataset_root / "coco_annotations"
    )
    ensure_exists(output_dir)

    bpe_path = (SAM3_ROOT / BPE_RELATIVE_PATH).resolve()
    if not bpe_path.exists():
        raise FileNotFoundError(
            f"Missing BPE file at {bpe_path}. Please ensure assets are installed."
        )

    limits = {
        "train": args.limit_train_images,
        "val": args.limit_val_images,
        "test": args.limit_test_images,
    }
    coco_paths = create_coco_annotations(
        dataset_root, coco_dir, force=args.force_coco, limits=limits
    )

    config_text = render_config_text(
        dataset_root=dataset_root,
        output_dir=output_dir,
        bpe_path=bpe_path,
        coco_paths=coco_paths,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr_scale=args.lr_scale,
        seed=args.seed,
        run_name=args.run_name,
    )
    config_path = (
        SAM3_CONFIG_DIR / "omnicrack_autogen" / f"{args.run_name}.yaml"
    ).resolve()
    config_rel_path = write_config_file(config_path, config_text)

    checkpoint_path = output_dir / "checkpoints" / "checkpoint.pt"

    if not args.skip_train:
        launch_training(config_rel_path, args.num_gpus, args.num_nodes)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Expected checkpoint not found at {checkpoint_path}"
            )
    else:
        print("Skipping training as requested (--skip_train).")

    if args.skip_inference:
        print("Skipping inference as requested (--skip_inference).")
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot run inference without checkpoint: {checkpoint_path}"
        )

    run_inference(
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        img_size=args.img_size,
        score_threshold=args.score_threshold,
        splits=args.inference_splits,
        output_dir=output_dir,
        max_images=args.max_infer_images,
    )


if __name__ == "__main__":
    main()
