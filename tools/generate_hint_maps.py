"""Utility to derive LumiNet control hints from plain RGB images.

The script produces 6-channel hint tensors (RGB + bright-region mask repeated
across three channels) stored as `.npy` files, plus an optional CSV manifest for
`train_adapter.py`.
"""
import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    return sorted(p for p in root.iterdir() if p.suffix.lower() in exts)


def load_and_resize(path: Path, size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if size > 0:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


def compute_bright_mask(gray: np.ndarray, blur_ksize: int, percentile: float) -> np.ndarray:
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    thresh = np.percentile(blurred, percentile)
    mask = np.clip((blurred - thresh) / max(1e-6, 1.0 - thresh), 0.0, 1.0)
    return mask


def build_hint(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask3 = np.repeat(mask[..., None], 3, axis=2)
    hint = np.concatenate([rgb, mask3], axis=2)
    return hint


def save_hint(hint: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, hint.astype(np.float32))


def write_manifest(entries: Iterable[Tuple[Path, Path]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "hint", "prompt"])
        for img, hint in entries:
            writer.writerow([str(img), str(hint), ""])


def process_directory(
    image_dir: Path,
    output_dir: Path,
    resize: int,
    blur_ksize: int,
    percentile: float,
) -> List[Tuple[Path, Path]]:
    entries: List[Tuple[Path, Path]] = []
    for img_path in list_images(image_dir):
        rgb = load_and_resize(img_path, resize)
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mask = compute_bright_mask(gray, blur_ksize=blur_ksize, percentile=percentile)
        hint = build_hint(rgb, mask)
        hint_path = output_dir / f"{img_path.stem}_hint.npy"
        save_hint(hint, hint_path)
        entries.append((img_path, hint_path))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LumiNet hint maps from RGB images.")
    parser.add_argument("--images", type=str, required=True, help="Directory of training RGB images.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save hint npy files.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional CSV path to save image/hint pairs for train_adapter.py",
    )
    parser.add_argument("--resize", type=int, default=512, help="Resize square resolution for hints (default: 512).")
    parser.add_argument("--blur-ksize", type=int, default=11, help="Gaussian blur kernel size for bright mask.")
    parser.add_argument(
        "--bright-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold to decide what counts as a bright region (default: 95).",
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = process_directory(
        image_dir=image_dir,
        output_dir=output_dir,
        resize=args.resize,
        blur_ksize=args.blur_ksize,
        percentile=args.bright_percentile,
    )

    if args.manifest:
        write_manifest(entries, Path(args.manifest))
        print(f"Wrote manifest with {len(entries)} entries to {args.manifest}")
    else:
        print(f"Generated {len(entries)} hint maps in {output_dir}")


if __name__ == "__main__":
    main()
