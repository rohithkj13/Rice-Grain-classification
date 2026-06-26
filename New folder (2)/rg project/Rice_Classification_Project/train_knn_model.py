"""
KNN Rice Classification Model
Trains a K-Nearest Neighbors classifier using CNN features for rice grain classification
"""

import os
import numpy as np
from pathlib import Path
import PIL
import pickle
import json

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = 'Rice_Image_Dataset'
MODEL_DIR = 'models'
IMAGE_SIZE = 224
BATCH_SIZE = 64
K_NEIGHBORS = 11   # Larger K = more stable/consistent predictions
MAX_IMAGES_PER_CLASS = 1000  # Keep KNN refresh fast; the CNN is now the primary signal.

RICE_CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Create models directory
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# LOAD AND PREPROCESS IMAGES
# ============================================================================

def load_images_from_folder(folder_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Load images from a folder"""
    images = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return images
    
    files = [f for f in os.listdir(folder_path) 
             if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"  Loading {len(files)} images from {folder_path}...")
    
    for filename in files[:MAX_IMAGES_PER_CLASS]:  # Use 3000 images per class for robust real-world accuracy
        try:
            img_path = os.path.join(folder_path, filename)
            img = PIL.Image.open(img_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(target_size)
            images.append(np.array(img))
        except Exception as e:
            print(f"  [SKIP] Error loading {filename}: {str(e)}")
            continue
    
    return np.array(images)

# ============================================================================
# EXTRACT CNN FEATURES
# ============================================================================

def extract_features_from_images(images):
    """Extract features using pre-trained MobileNetV2"""
    print("\n[STEP] Extracting CNN features from images...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Preprocess images for MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    preprocessed = preprocess_input(images.astype('float32'))
    
    # Extract features
    features = base_model.predict(preprocessed, batch_size=BATCH_SIZE, verbose=1)
    
    print(f"[OK] Extracted features shape: {features.shape}")
    return features

# ============================================================================
# TRAIN KNN MODEL
# ============================================================================

def train_knn_model():
    """Train KNN classifier on rice dataset - memory-efficient class-by-class feature extraction"""
    
    print("="*70)
    print("KNN RICE GRAIN CLASSIFICATION - TRAINING")
    print("="*70)
    
    # Load pretrained MobileNetV2 once (reused across all classes)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    print("\n[INIT] Loading MobileNetV2 feature extractor...")
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )
    base_model.trainable = False
    print("[OK] Feature extractor ready")
    
    # Process one class at a time (avoids loading all 15,000 images into RAM)
    print("\n[STEP 1+2] LOADING IMAGES & EXTRACTING FEATURES (class-by-class)")
    print("-"*70)
    
    all_features = []
    y_train = []
    
    for class_name in RICE_CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        print(f"\nProcessing {class_name}...")
        images = load_images_from_folder(class_path)
        
        if len(images) == 0:
            print(f"[ERROR] No images found for {class_name}")
            continue
        
        print(f"  Extracting features for {len(images)} images...")
        preprocessed = preprocess_input(images.astype('float32'))
        features = base_model.predict(preprocessed, batch_size=BATCH_SIZE, verbose=0)
        
        all_features.append(features)
        y_train.extend([class_name] * len(images))
        print(f"  [OK] Features shape: {features.shape}")
        
        # Free memory immediately after processing each class
        del images, preprocessed, features
    
    X_train_features = np.vstack(all_features)
    del all_features
    y_train = np.array(y_train)
    
    print(f"\n[OK] Total training samples: {len(X_train_features)}")
    print(f"[OK] Feature vector size: {X_train_features.shape[1]}")
    
    # Encode labels
    print("\n[STEP 3] ENCODING LABELS")
    print("-"*70)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    print(f"[OK] Classes: {label_encoder.classes_}")
    
    # Train KNN
    print("\n[STEP 4] TRAINING KNN CLASSIFIER")
    print("-"*70)
    print(f"Using K={K_NEIGHBORS} neighbors...")
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, n_jobs=-1, metric='euclidean')
    knn.fit(X_train_features, y_train_encoded)
    print(f"[OK] KNN model trained!")
    
    # Evaluate on training data
    print("\n[STEP 5] EVALUATING MODEL")
    print("-"*70)
    y_pred = knn.predict(X_train_features)
    accuracy = accuracy_score(y_train_encoded, y_pred)
    print(f"[OK] Training Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_train_encoded, y_pred, 
                                target_names=label_encoder.classes_))
    
    # Save models
    print("\n[STEP 6] SAVING MODELS")
    print("-"*70)
    
    # Save KNN model
    knn_path = os.path.join(MODEL_DIR, 'rice_knn_classifier.pkl')
    with open(knn_path, 'wb') as f:
        pickle.dump(knn, f)
    print(f"[OK] KNN model saved to: {knn_path}")
    
    # Save label encoder
    encoder_path = os.path.join(MODEL_DIR, 'rice_label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"[OK] Label encoder saved to: {encoder_path}")
    
    # Save feature extractor (base model)
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )
    feature_extractor_path = os.path.join(MODEL_DIR, 'rice_feature_extractor.h5')
    base_model.save(feature_extractor_path)
    print(f"[OK] Feature extractor saved to: {feature_extractor_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'KNN',
        'n_neighbors': K_NEIGHBORS,
        'accuracy': float(accuracy),
        'image_size': IMAGE_SIZE,
        'classes': RICE_CLASSES,
        'training_samples': int(len(X_train_features))
    }
    
    metadata_path = os.path.join(MODEL_DIR, 'rice_knn_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved to: {metadata_path}")
    
    print("\n" + "="*70)
    print("KNN TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel Files Generated:")
    print(f"  1. {knn_path}")
    print(f"  2. {encoder_path}")
    print(f"  3. {feature_extractor_path}")
    print(f"  4. {metadata_path}")
    
    return knn, label_encoder

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        knn_model, label_enc = train_knn_model()
        print("\n[SUCCESS] KNN model training complete!")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
