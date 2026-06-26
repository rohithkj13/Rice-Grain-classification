"""
Dataset Cleaning & Smart Reduction Module
==========================================
Reduces ~15,000 images/class to a high-quality 3,000–5,000 subset by:
  1. Removing blurry images (Laplacian variance threshold)
  2. Removing near-duplicate images (perceptual hash)
  3. Removing low-contrast / near-solid images (std-dev threshold)
  4. Stratified random sampling to hit the target count per class
"""

from __future__ import annotations

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# ── Thresholds ────────────────────────────────────────────────────────────────
BLUR_THRESHOLD        = 80.0    # Laplacian variance; below → blurry
CONTRAST_THRESHOLD    = 8.0     # Pixel std-dev;      below → near-solid
PHASH_BITS            = 16      # Perceptual hash grid size
DUPLICATE_HAMMING     = 6       # Max bit-diff to call two images duplicates
TARGET_PER_CLASS      = 4000    # Images to keep per class (adjust as needed)
RANDOM_SEED           = 42


# ── Perceptual hash helpers ───────────────────────────────────────────────────

def _phash(image_bgr: np.ndarray, size: int = PHASH_BITS) -> int:
    """Compute a simple average-hash integer for near-duplicate detection."""
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean  = small.mean()
    bits  = (small > mean).flatten()
    # Pack bits into a Python int
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    """Hamming distance between two hash integers."""
    return bin(a ^ b).count('1')


# ── Per-image quality checks ──────────────────────────────────────────────────

def is_blurry(image_bgr: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def is_low_contrast(image_bgr: np.ndarray, threshold: float = CONTRAST_THRESHOLD) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray)) < threshold


# ── Main cleaner ─────────────────────────────────────────────────────────────

class DatasetCleaner:
    """
    Scans a dataset folder (one sub-folder per class), removes bad images,
    deduplicates, and writes a clean balanced subset to an output folder.

    Expected input layout:
        dataset_root/
            Arborio/   *.jpg  *.png
            Basmati/   ...
            Ipsala/    ...
            Jasmine/   ...
            Karacadag/ ...

    Output layout mirrors the input.
    """

    def __init__(
        self,
        dataset_root: str,
        output_root: str,
        target_per_class: int = TARGET_PER_CLASS,
        blur_threshold: float = BLUR_THRESHOLD,
        contrast_threshold: float = CONTRAST_THRESHOLD,
        duplicate_hamming: int = DUPLICATE_HAMMING,
        seed: int = RANDOM_SEED,
    ):
        self.dataset_root      = Path(dataset_root)
        self.output_root       = Path(output_root)
        self.target_per_class  = target_per_class
        self.blur_threshold    = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.duplicate_hamming = duplicate_hamming
        self.seed              = seed
        self.stats: Dict[str, Dict] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _image_files(self, folder: Path) -> List[Path]:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        return [p for p in folder.iterdir() if p.suffix.lower() in exts]

    def _quality_filter(self, paths: List[Path]) -> Tuple[List[Path], Dict]:
        """Return paths that pass blur + contrast checks."""
        passed, n_blur, n_contrast = [], 0, 0
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if is_blurry(img, self.blur_threshold):
                n_blur += 1
                continue
            if is_low_contrast(img, self.contrast_threshold):
                n_contrast += 1
                continue
            passed.append(p)
        return passed, {'removed_blur': n_blur, 'removed_contrast': n_contrast}

    def _deduplicate(self, paths: List[Path]) -> Tuple[List[Path], int]:
        """Remove near-duplicate images using perceptual hashing."""
        hashes: List[Tuple[int, Path]] = []
        unique: List[Path] = []
        n_dup = 0

        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h = _phash(img)
            # Check against all stored hashes
            is_dup = any(_hamming(h, stored_h) <= self.duplicate_hamming
                         for stored_h, _ in hashes)
            if is_dup:
                n_dup += 1
            else:
                hashes.append((h, p))
                unique.append(p)

        return unique, n_dup

    def _stratified_sample(self, paths: List[Path], n: int) -> List[Path]:
        """Randomly sample n paths (or all if fewer available)."""
        if len(paths) <= n:
            return paths
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(paths), size=n, replace=False)
        return [paths[i] for i in sorted(indices)]

    # ── Public API ────────────────────────────────────────────────────────────

    def clean_class(self, class_folder: Path) -> List[Path]:
        """
        Full cleaning pipeline for one class folder.
        Returns the final list of selected image paths.
        """
        class_name = class_folder.name
        all_paths  = self._image_files(class_folder)
        n_original = len(all_paths)

        print(f"\n  [{class_name}] {n_original} images found")

        # Step 1 – quality filter
        passed, q_stats = self._quality_filter(all_paths)
        print(f"    After quality filter : {len(passed):>5}  "
              f"(removed blur={q_stats['removed_blur']}, "
              f"contrast={q_stats['removed_contrast']})")

        # Step 2 – deduplication
        unique, n_dup = self._deduplicate(passed)
        print(f"    After deduplication  : {len(unique):>5}  (removed {n_dup} duplicates)")

        # Step 3 – stratified sampling
        selected = self._stratified_sample(unique, self.target_per_class)
        print(f"    Final selected       : {len(selected):>5}")

        self.stats[class_name] = {
            'original'         : n_original,
            'after_quality'    : len(passed),
            'after_dedup'      : len(unique),
            'final_selected'   : len(selected),
            **q_stats,
            'removed_duplicates': n_dup,
        }

        return selected

    def run(self) -> Dict[str, Dict]:
        """
        Run the full cleaning pipeline over all class folders.
        Copies selected images to self.output_root preserving class structure.
        Returns per-class statistics.
        """
        class_folders = sorted([d for d in self.dataset_root.iterdir() if d.is_dir()])
        if not class_folders:
            raise FileNotFoundError(f"No class sub-folders found in {self.dataset_root}")

        print(f"\n{'='*60}")
        print(f"  Dataset Cleaning & Reduction Pipeline")
        print(f"  Source : {self.dataset_root}")
        print(f"  Output : {self.output_root}")
        print(f"  Target : {self.target_per_class} images/class")
        print(f"{'='*60}")

        total_copied = 0
        for class_folder in class_folders:
            selected = self.clean_class(class_folder)

            # Copy to output
            out_class_dir = self.output_root / class_folder.name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            for src in selected:
                shutil.copy2(src, out_class_dir / src.name)
            total_copied += len(selected)

        print(f"\n{'='*60}")
        print(f"  Cleaning complete. Total images copied: {total_copied}")
        print(f"{'='*60}\n")

        self._print_summary()
        return self.stats

    def _print_summary(self):
        print(f"\n{'Class':<15} {'Original':>10} {'Quality':>10} "
              f"{'Dedup':>10} {'Final':>10}")
        print("-" * 60)
        for cls, s in self.stats.items():
            print(f"{cls:<15} {s['original']:>10} {s['after_quality']:>10} "
                  f"{s['after_dedup']:>10} {s['final_selected']:>10}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rice Dataset Cleaner')
    parser.add_argument('--input',   required=True,  help='Path to raw dataset root')
    parser.add_argument('--output',  required=True,  help='Path to cleaned output root')
    parser.add_argument('--target',  type=int, default=TARGET_PER_CLASS,
                        help=f'Images per class (default: {TARGET_PER_CLASS})')
    parser.add_argument('--blur',    type=float, default=BLUR_THRESHOLD,
                        help=f'Blur threshold (default: {BLUR_THRESHOLD})')
    args = parser.parse_args()

    cleaner = DatasetCleaner(
        dataset_root=args.input,
        output_root=args.output,
        target_per_class=args.target,
        blur_threshold=args.blur,
    )
    cleaner.run()
