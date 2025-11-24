"""
Training script for LumiNet control adapters.

You can train from either a CSV manifest or directly from an RGB image folder:
- Manifest columns: `image` (RGB path), `hint` (6채널 힌트맵 경로), `prompt` (선택).
- Folder-only mode: pass `--train-images /path/to/rgb_dir` and the script will create
  6채널 힌트를 온더플라이로 생성합니다.

Example usage:
    python train_adapter.py \
        --config models/cldm_v21_LumiNet.yaml \
        --train-manifest /path/to/train.csv \
        --val-manifest /path/to/val.csv \
        --pretrained-ckpt /path/to/luminet.ckpt \
        --adapter-init /path/to/adapter_only.ckpt \
        --logdir logs/adapter_run
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from cldm.model import create_model, load_state_dict


def _default_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ]
    )


def _load_hint(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        hint = np.load(path)
        if hint.ndim == 2:
            hint = hint[:, :, None]
        return hint
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img) / 127.5 - 1.0


class LightingAdapterDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        hint_channels: int,
        manifest: Optional[str] = None,
        image_dir: Optional[str] = None,
        auto_bright_percentile: float = 95.0,
        auto_blur_ksize: int = 11,
    ):
        super().__init__()
        self.entries = self._read_entries(manifest, image_dir)
        self.image_transform = _default_image_transform(image_size)
        self.hint_channels = hint_channels
        self.image_size = image_size
        self.auto_bright_percentile = auto_bright_percentile
        self.auto_blur_ksize = auto_blur_ksize if auto_blur_ksize % 2 == 1 else auto_blur_ksize + 1

    @staticmethod
    def _read_manifest(path: str) -> List[dict]:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"image", "hint"}
            missing = required - set(reader.fieldnames or {})
            if missing:
                raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
            return [row for row in reader]

    @staticmethod
    def _read_images(image_dir: str) -> List[dict]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        paths = [p for p in sorted(Path(image_dir).iterdir()) if p.suffix.lower() in exts]
        if not paths:
            raise ValueError(f"No images found in {image_dir}")
        return [{"image": str(p), "hint": None, "prompt": ""} for p in paths]

    def _read_entries(self, manifest: Optional[str], image_dir: Optional[str]) -> List[dict]:
        if manifest:
            return self._read_manifest(manifest)
        if image_dir:
            return self._read_images(image_dir)
        raise ValueError("Either a manifest or an image directory must be provided.")

    def _normalize_hint(self, hint: np.ndarray) -> torch.Tensor:
        if hint.ndim == 2:
            hint = hint[:, :, None]
        if hint.shape[2] > self.hint_channels:
            hint = hint[:, :, : self.hint_channels]
        elif hint.shape[2] < self.hint_channels:
            pad = self.hint_channels - hint.shape[2]
            hint = np.concatenate([hint, np.zeros((*hint.shape[:2], pad), dtype=hint.dtype)], axis=2)

        tensor = torch.from_numpy(hint).float()
        max_val, min_val = tensor.max(), tensor.min()
        if max_val > 1.0 or min_val < -1.0:
            tensor = tensor / 127.5 - 1.0
        elif 0.0 <= min_val <= max_val <= 1.0:
            tensor = tensor * 2.0 - 1.0
        return tensor

    def _resize_hint(self, hint: torch.Tensor) -> torch.Tensor:
        """
        Resize hints to match the training image resolution.

        Expects a tensor shaped (C, H, W) in [-1, 1].
        """
        if hint.shape[1] == self.image_size and hint.shape[2] == self.image_size:
            return hint
        hint = hint.unsqueeze(0)
        resized = F.interpolate(hint, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return resized.squeeze(0)

    def _auto_hint_from_rgb(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Derive a 6-channel hint (RGB + bright-region mask x3) from an RGB image tensor.

        Args:
            rgb_tensor: Tensor shaped (H, W, 3) in [-1, 1] after resizing.
        """

        rgb_chw = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
        rgb_01 = (rgb_chw + 1.0) / 2.0
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_tensor.device).view(1, 3, 1, 1)
        gray = (rgb_01 * weights).sum(dim=1, keepdim=True)

        blurred = TF.gaussian_blur(gray, kernel_size=self.auto_blur_ksize)
        thresh = torch.quantile(blurred.flatten(), self.auto_bright_percentile / 100.0)
        mask = torch.clamp((blurred - thresh) / torch.clamp(1.0 - thresh, min=1e-6), 0.0, 1.0)
        mask = mask * 2.0 - 1.0
        mask3 = mask.expand(-1, 3, -1, -1)

        hint = torch.cat([rgb_chw, mask3], dim=1)
        return hint.squeeze(0).permute(1, 2, 0)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        with Image.open(entry["image"]) as img:
            rgb = img.convert("RGB")
            rgb_tensor = self.image_transform(rgb)  # CxHxW in [-1, 1]
            rgb_tensor = rgb_tensor.permute(1, 2, 0)  # HxWxC to match ControlLDM expectation

        if entry["hint"]:
            hint_arr = _load_hint(entry["hint"])
            hint_tensor = self._resize_hint(self._normalize_hint(hint_arr).permute(2, 0, 1)).permute(1, 2, 0)
        else:
            hint_tensor = self._auto_hint_from_rgb(rgb_tensor)

        prompt = entry.get("prompt", "")
        return {"jpg": rgb_tensor, "hint": hint_tensor, "txt": prompt}


class LightingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size: int,
        hint_channels: int,
        batch_size: int,
        num_workers: int,
        train_manifest: Optional[str] = None,
        val_manifest: Optional[str] = None,
        train_images: Optional[str] = None,
        val_images: Optional[str] = None,
        auto_bright_percentile: float = 95.0,
        auto_blur_ksize: int = 11,
    ):
        super().__init__()
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.train_images = train_images
        self.val_images = val_images
        self.image_size = image_size
        self.hint_channels = hint_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.auto_bright_percentile = auto_bright_percentile
        self.auto_blur_ksize = auto_blur_ksize

    def setup(self, stage: Optional[str] = None):
        self.train_set = LightingAdapterDataset(
            manifest=self.train_manifest,
            image_dir=self.train_images,
            image_size=self.image_size,
            hint_channels=self.hint_channels,
            auto_bright_percentile=self.auto_bright_percentile,
            auto_blur_ksize=self.auto_blur_ksize,
        )
        self.val_set = None
        if self.val_manifest or self.val_images:
            self.val_set = LightingAdapterDataset(
                manifest=self.val_manifest,
                image_dir=self.val_images,
                image_size=self.image_size,
                hint_channels=self.hint_channels,
                auto_bright_percentile=self.auto_bright_percentile,
                auto_blur_ksize=self.auto_blur_ksize,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_set is None:
            return []
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def _freeze_for_adapter_training(model, train_cross_attention: bool):
    for module in [model.first_stage_model, model.cond_stage_model, model.model]:
        for p in module.parameters():
            p.requires_grad = False
    for p in model.control_model.parameters():
        p.requires_grad = True

    model.crossattn_start = bool(train_cross_attention)
    model.crossattn_mid = bool(train_cross_attention)
    model.crossattn = bool(train_cross_attention)


def _build_model(args) -> torch.nn.Module:
    model = create_model(args.config)
    if args.pretrained_ckpt:
        state_dict = load_state_dict(args.pretrained_ckpt, location="cpu")
        model.load_state_dict(state_dict, strict=False)
    if args.adapter_init:
        adapter_state = load_state_dict(args.adapter_init, location="cpu")
        model.control_model.load_state_dict(adapter_state, strict=False)

    model.learning_rate = args.learning_rate
    model.control_scales = [args.control_scale] * 13
    _freeze_for_adapter_training(model, args.train_cross_attention)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LumiNet lighting adapter")
    parser.add_argument("--config", type=str, default="models/cldm_v21_LumiNet.yaml", help="Model config yaml")
    parser.add_argument("--train-manifest", type=str, default=None, help="CSV with image, hint, prompt columns")
    parser.add_argument("--val-manifest", type=str, default=None, help="Optional CSV for validation")
    parser.add_argument("--train-images", type=str, default=None, help="Folder of RGB images (auto-generate hints).")
    parser.add_argument("--val-images", type=str, default=None, help="Optional folder of RGB images for validation.")
    parser.add_argument("--pretrained-ckpt", type=str, default=None, help="Full LumiNet checkpoint to initialize")
    parser.add_argument("--adapter-init", type=str, default=None, help="Optional adapter-only weights for warm start")
    parser.add_argument("--logdir", type=str, default="logs/adapter", help="Checkpoint and log directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--precision", type=int, choices=[16, 32], default=16)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--train-cross-attention", action="store_true", help="Also fine-tune UNet cross-attention")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from a checkpoint")
    parser.add_argument("--auto-bright-percentile", type=float, default=95.0, help="Percentile for auto bright mask.")
    parser.add_argument("--auto-blur-ksize", type=int, default=11, help="Gaussian blur kernel for auto hints.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.train_manifest and not args.train_images:
        raise ValueError("At least one of --train-manifest or --train-images must be provided.")

    config = OmegaConf.load(args.config)

    data = LightingDataModule(
        image_size=args.image_size,
        hint_channels=config.model.params.control_stage_config.params.hint_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        train_images=args.train_images,
        val_images=args.val_images,
        auto_bright_percentile=args.auto_bright_percentile,
        auto_blur_ksize=args.auto_blur_ksize,
    )

    model = _build_model(args)

    has_val = bool(args.val_manifest or args.val_images)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.logdir,
        save_top_k=3,
        monitor="val/loss_simple_ema" if has_val else "train/loss",
        filename="adapter-{epoch:02d}-{global_step:06d}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=args.precision,
        default_root_dir=args.logdir,
        resume_from_checkpoint=args.resume_from,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=True,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
