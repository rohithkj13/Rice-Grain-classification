"""
Quick Rice Model Training Script
Trains a CNN to identify rice varieties from the dataset
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import json

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = 'Rice_Image_Dataset'
MODEL_PATH = 'models/rice_classifier.h5'
MODEL_METADATA_PATH = 'models/rice_classifier_metadata.json'
OUTPUT_DIR = 'outputs'
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

RICE_CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# ============================================================================
# DATA LOADING
# ============================================================================

def load_images_from_folder(folder_path, class_name, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Load images from a folder"""
    images = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return images
    
    files = [f for f in os.listdir(folder_path) 
             if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"  Found {len(files)} images in {class_name}")
    
    for filename in files[:100]:  # Limit to 100 per class for quick training
        try:
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(target_size)
            
            # Convert to array
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
        except Exception as e:
            print(f"    Error loading {filename}: {str(e)}")
    
    return images

def prepare_dataset():
    """Load and prepare the dataset"""
    print("\n" + "="*60)
    print("📊 Loading Dataset")
    print("="*60)
    
    X_data = []
    y_data = []
    
    for idx, rice_type in enumerate(RICE_CLASSES):
        folder_path = os.path.join(DATASET_PATH, rice_type)
        print(f"\n{idx+1}. Loading {rice_type}...")
        
        images = load_images_from_folder(folder_path, rice_type)
        
        if len(images) == 0:
            print(f"   ⚠️ No images found for {rice_type}")
            continue
        
        X_data.extend(images)
        y_data.extend([idx] * len(images))
    
    if len(X_data) == 0:
        print("\n❌ No images found in dataset!")
        print(f"Expected dataset at: {os.path.abspath(DATASET_PATH)}")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data)
    
    print(f"\n✓ Dataset loaded: {X.shape[0]} images, {len(RICE_CLASSES)} classes")
    
    # Stratified split: 70% train, 20% validation, 10% test
    X_train, X_temp, y_train_idx, y_temp_idx = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
    X_val, X_test, y_val_idx, y_test_idx = train_test_split(
        X_temp,
        y_temp_idx,
        test_size=(1.0 / 3.0),
        random_state=42,
        stratify=y_temp_idx
    )

    y_train = keras.utils.to_categorical(y_train_idx, num_classes=len(RICE_CLASSES))
    y_val = keras.utils.to_categorical(y_val_idx, num_classes=len(RICE_CLASSES))
    y_test = keras.utils.to_categorical(y_test_idx, num_classes=len(RICE_CLASSES))
    
    print(f"\n Split:")
    print(f"  Train: {X_train.shape[0]} images")
    print(f"  Val:   {X_val.shape[0]} images")
    print(f"  Test:  {X_test.shape[0]} images")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(num_classes=5):
    """Build transfer learning model"""
    print("\n" + "="*60)
    print("🔨 Building Model")
    print("="*60)
    
    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    print(f"✓ Base model: MobileNetV2 ({len(base_model.layers)} layers frozen)")
    
    # Build custom classifier
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model built and compiled")
    
    return model

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_data, val_data):
    """Train the model"""
    print("\n" + "="*60)
    print("🚀 Training Model")
    print("="*60)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Data augmentation
    augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    print(f"\n📈 Training for {EPOCHS} epochs...")
    
    history = model.fit(
        augmentation.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    print("\n✓ Training completed!")
    
    return history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_data):
    """Evaluate model on test set"""
    print("\n" + "="*60)
    print("📊 Model Evaluation")
    print("="*60)
    
    X_test, y_test = test_data
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✓ Test Loss: {loss:.4f}")
    print(f"✓ Test Accuracy: {accuracy*100:.2f}%")
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i, rice_type in enumerate(RICE_CLASSES):
        mask = true_classes == i
        if mask.sum() > 0:
            class_acc = (pred_classes[mask] == true_classes[mask]).mean()
            print(f"  {rice_type}: {class_acc*100:.2f}%")
    
    return accuracy

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🍚 Rice Classification Model Training")
    print("="*60)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    dataset = prepare_dataset()
    if dataset is None:
        return
    
    train_data, val_data, test_data = dataset
    
    # Build model
    model = build_model(len(RICE_CLASSES))
    
    # Train
    history = train_model(model, train_data, val_data)
    
    # Evaluate
    accuracy = evaluate_model(model, test_data)
    
    # Save model
    print("\n" + "="*60)
    print("💾 Saving Model")
    print("="*60)
    
    model.save(MODEL_PATH)
    print(f"✓ Model saved to: {MODEL_PATH}")
    
    # Save metadata
    metadata = {
        'class_names': RICE_CLASSES,
        'classes': RICE_CLASSES,
        'class_to_index': {name: idx for idx, name in enumerate(RICE_CLASSES)},
        'image_size': IMAGE_SIZE,
        'input_shape': [IMAGE_SIZE, IMAGE_SIZE, 3],
        'color_format': 'RGB',
        'normalization': 'divide_by_255',
        'test_accuracy': float(accuracy),
        'model_type': 'MobileNetV2 Transfer Learning',
        'epochs_trained': EPOCHS
    }
    
    with open(os.path.join(OUTPUT_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    with open(MODEL_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\n📝 Model ready for predictions at: {MODEL_PATH}")
    print("🌐 Restart the Flask app to use the trained model")
    print("\n")

if __name__ == '__main__':
    main()
