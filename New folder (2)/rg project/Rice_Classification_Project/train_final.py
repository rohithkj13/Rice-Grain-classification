"""
Step 2 — Training Pipeline
============================
- EfficientNetB0 transfer learning (better than MobileNetV2 for fine-grained)
- 6 rice classes + non_rice = 6 total
- Consistent preprocessing: EfficientNet expects [0,255] uint8 (no manual scaling)
- Two-phase training: frozen → fine-tune
- Class weights to handle any remaining imbalance
- Saves model + metadata for inference

Usage:
    python train_final.py --dataset dataset_clean
"""

from __future__ import annotations
import argparse, json, os, warnings
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_NAMES     = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag', 'non_rice']
IMG_SIZE        = 224
BATCH_SIZE      = 16          # small dataset → small batch
FROZEN_EPOCHS   = 20
FINETUNE_EPOCHS = 20
FROZEN_LR       = 1e-3
FINETUNE_LR     = 1e-5
SEED            = 42
MODEL_DIR       = Path('models')
AUTOTUNE        = tf.data.AUTOTUNE


# ── tf.data pipeline ──────────────────────────────────────────────────────────

def _load(path: tf.Tensor, label: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    # EfficientNetB0 expects raw [0,255] — its own rescaling is inside the model
    return img, label


def _augment(img: tf.Tensor, label: tf.Tensor):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.20)
    img = tf.image.random_contrast(img, 0.75, 1.25)
    img = tf.image.random_saturation(img, 0.75, 1.25)
    img = tf.image.random_hue(img, 0.05)
    # random crop (zoom-in effect)
    padded = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 30, IMG_SIZE + 30)
    img    = tf.image.random_crop(padded, [IMG_SIZE, IMG_SIZE, 3])
    img    = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def make_dataset(root: Path, split: str, augment: bool) -> tf.data.Dataset:
    split_dir = root / split
    paths, labels = [], []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.iterdir():
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                paths.append(str(p))
                labels.append(cls_idx)

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32))
    )
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE), labels


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')
    # EfficientNetB0 has its own rescaling layer internally
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.50)(x)
    x   = layers.Dense(256, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inp, out, name='rice_efficientnet')
    return model, base


def get_callbacks(ckpt: str) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            ckpt, monitor='val_accuracy',
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.4,
            patience=4, min_lr=1e-8, verbose=1),
    ]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: keras.Model, ds: tf.data.Dataset, true_labels: list) -> dict:
    probs = model.predict(ds, verbose=1)
    preds = np.argmax(probs, axis=1)
    acc   = float(np.mean(preds == np.array(true_labels)))
    report = classification_report(
        true_labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    cm = confusion_matrix(true_labels, preds)
    print(f"\n  Accuracy: {acc*100:.2f}%")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    return {'accuracy': acc, 'confusion_matrix': cm.tolist()}


# ── Main ──────────────────────────────────────────────────────────────────────

def train(dataset_root: str) -> None:
    tf.random.set_seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)

    root = Path(dataset_root)
    print("\n=== Rice Classifier Training (EfficientNetB0) ===")

    train_ds, y_train = make_dataset(root, 'train', augment=True)
    val_ds,   y_val   = make_dataset(root, 'val',   augment=False)
    test_ds,  y_test  = make_dataset(root, 'test',  augment=False)

    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    # Class weights
    cw = compute_class_weight('balanced',
                              classes=np.arange(len(CLASS_NAMES)),
                              y=np.array(y_train))
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    print("  Class weights:", {CLASS_NAMES[i]: round(v, 2)
                                for i, v in class_weights.items()})

    model, base = build_model(len(CLASS_NAMES))
    ckpt = str(MODEL_DIR / 'rice_classifier.h5')

    # Phase 1 — frozen backbone
    print("\n--- Phase 1: Frozen backbone ---")
    model.compile(
        optimizer=keras.optimizers.Adam(FROZEN_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds,
              epochs=FROZEN_EPOCHS,
              callbacks=get_callbacks(ckpt),
              class_weight=class_weights, verbose=1)

    # Phase 2 — fine-tune top 50% of base
    print("\n--- Phase 2: Fine-tuning ---")
    base.trainable = True
    freeze_until = int(len(base.layers) * 0.50)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(FINETUNE_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds,
              epochs=FROZEN_EPOCHS + FINETUNE_EPOCHS,
              initial_epoch=FROZEN_EPOCHS,
              callbacks=get_callbacks(ckpt),
              class_weight=class_weights, verbose=1)

    # Load best weights
    model = keras.models.load_model(ckpt)

    print("\n--- Test Set Evaluation ---")
    metrics = evaluate(model, test_ds, y_test)

    # Save metadata
    meta = {
        'class_names'   : CLASS_NAMES,
        'classes'       : CLASS_NAMES,
        'class_to_index': {n: i for i, n in enumerate(CLASS_NAMES)},
        'image_size'    : IMG_SIZE,
        'input_shape'   : [IMG_SIZE, IMG_SIZE, 3],
        'color_format'  : 'RGB',
        'normalization' : 'efficientnet_internal',
        'model_name'    : 'EfficientNetB0',
        'non_rice_index': CLASS_NAMES.index('non_rice'),
        'test_accuracy' : metrics['accuracy'],
    }
    (MODEL_DIR / 'rice_classifier_metadata.json').write_text(
        json.dumps(meta, indent=2))
    print(f"\n  Model  → {ckpt}")
    print(f"  Meta   → models/rice_classifier_metadata.json")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset_clean')
    args = ap.parse_args()
    train(args.dataset)
