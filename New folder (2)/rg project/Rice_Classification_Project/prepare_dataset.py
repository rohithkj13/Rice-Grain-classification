"""
Step 1 — Dataset Preparation
=============================
- Removes blurry / corrupt / duplicate images
- Balances all rice classes to TARGET_PER_CLASS images
- Creates a 'non_rice' class from random internet-free placeholder images
  (or copies from a user-supplied folder)
- Splits into train / val / test  (70 / 15 / 15)
- Saves split manifests as JSON for reproducibility

Usage:
    python prepare_dataset.py --src Rice_Image_Dataset --out dataset_clean
    python prepare_dataset.py --src Rice_Image_Dataset --out dataset_clean --nonrice path/to/nonrice_images
"""

from __future__ import annotations
import argparse, hashlib, json, random, shutil
from pathlib import Path
import cv2, numpy as np
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_NAMES       = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag', 'non_rice']
TARGET_PER_CLASS  = 200   # images kept per class after cleaning
BLUR_THRESH       = 60.0  # Laplacian variance — below = blurry
SEED              = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def is_blurry(path: Path) -> bool:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    return float(cv2.Laplacian(img, cv2.CV_64F).var()) < BLUR_THRESH

def is_corrupt(path: Path) -> bool:
    img = cv2.imread(str(path))
    return img is None

def clean_class(src_dir: Path, target: int, seed: int) -> list[Path]:
    """Return up to `target` clean, unique image paths from src_dir."""
    exts  = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]
    random.seed(seed)
    random.shuffle(files)

    seen_hashes, good = set(), []
    for p in files:
        if is_corrupt(p) or is_blurry(p):
            continue
        h = file_hash(p)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        good.append(p)
        if len(good) >= target:
            break

    print(f"  {src_dir.name:<14}: {len(files):>5} found → {len(good):>3} kept")
    return good

def generate_nonrice_synthetic(out_dir: Path, n: int) -> None:
    """
    Generate simple synthetic non-rice images (solid colours + noise).
    Replace this with real non-rice images for best results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for i in range(n):
        # random coloured background + gaussian noise
        colour = rng.integers(30, 220, size=3).tolist()
        img    = np.full((224, 224, 3), colour, dtype=np.uint8)
        noise  = rng.integers(-40, 40, img.shape, dtype=np.int16)
        img    = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"nonrice_{i:04d}.jpg"), img)
    print(f"  {'non_rice':<14}: {n} synthetic images generated")

# ── Main ──────────────────────────────────────────────────────────────────────

def prepare(src: str, out: str, nonrice_src: str | None) -> None:
    src_root = Path(src)
    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n=== Dataset Preparation ===")
    all_paths, all_labels = [], []

    # Rice classes
    for cls in CLASS_NAMES[:-1]:
        cls_dir = src_root / cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} not found, skipping")
            continue
        good = clean_class(cls_dir, TARGET_PER_CLASS, SEED)
        label = CLASS_NAMES.index(cls)
        all_paths  += good
        all_labels += [label] * len(good)

    # Non-rice class
    nr_dir = out_root / 'non_rice_raw'
    if nonrice_src:
        shutil.copytree(nonrice_src, nr_dir, dirs_exist_ok=True)
        good_nr = clean_class(nr_dir, TARGET_PER_CLASS, SEED)
    else:
        generate_nonrice_synthetic(nr_dir, TARGET_PER_CLASS)
        good_nr = list(nr_dir.glob('*.jpg'))[:TARGET_PER_CLASS]

    nr_label = CLASS_NAMES.index('non_rice')
    all_paths  += good_nr
    all_labels += [nr_label] * len(good_nr)

    # Stratified split 70 / 15 / 15
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=SEED)

    # Copy files into split folders
    for split_name, split_paths, split_labels in [
        ('train', X_train, y_train),
        ('val',   X_val,   y_val),
        ('test',  X_test,  y_test),
    ]:
        for p, lbl in zip(split_paths, split_labels):
            cls_name = CLASS_NAMES[lbl]
            dest_dir = out_root / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest_dir / Path(p).name)

    # Save manifest
    manifest = {
        'class_names': CLASS_NAMES,
        'train': len(X_train), 'val': len(X_val), 'test': len(X_test),
        'total': len(all_paths),
    }
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f"\n  Split → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    print(f"  Output: {out_root.resolve()}")
    print("  NOTE: Replace synthetic non_rice images with real ones for best accuracy.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src',     default='Rice_Image_Dataset')
    ap.add_argument('--out',     default='dataset_clean')
    ap.add_argument('--nonrice', default=None,
                    help='Optional path to real non-rice images folder')
    args = ap.parse_args()
    prepare(args.src, args.out, args.nonrice)
