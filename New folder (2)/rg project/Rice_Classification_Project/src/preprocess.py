"""
Image Preprocessing Module for Rice Classification
Handles dataset loading, resizing, normalization, and augmentation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


class ImagePreprocessor:
    """
    A class to handle image preprocessing for rice grain classification
    """
    
    def __init__(self, image_size=224):
        """
        Initialize the preprocessor with target image size
        
        Args:
            image_size (int): Target size for resized images (default: 224x224)
        """
        self.image_size = image_size
        self.classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        
    def load_dataset(self, dataset_path):
        """
        Load images from folder structure
        Each subfolder represents a rice class
        
        Args:
            dataset_path (str): Path to the dataset folder
            
        Returns:
            tuple: (images_array, labels_array, class_names)
        """
        images = []
        labels = []
        class_names = []
        
        dataset_path = Path(dataset_path)
        
        # Find all class folders
        class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        
        print(f"Found {len(class_folders)} classes")
        
        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            class_names.append(class_name)
            print(f"Loading class {class_idx + 1}/{len(class_folders)}: {class_name}")
            
            image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
            print(f"  Found {len(image_files)} images")
            
            for img_file in image_files:
                try:
                    # Read image
                    image = cv2.imread(str(img_file))
                    
                    if image is not None:
                        # Convert BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Resize image
                        image = cv2.resize(image, (self.image_size, self.image_size))
                        
                        images.append(image)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_file}: {str(e)}")
        
        # Convert to numpy arrays
        images_array = np.array(images, dtype=np.float32)
        labels_array = np.array(labels)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(images_array)}")
        print(f"Image shape: {images_array[0].shape if len(images_array) > 0 else 'N/A'}")
        
        return images_array, labels_array, class_names
    
    def normalize_images(self, images):
        """
        Normalize pixel values to range [0, 1]
        
        Args:
            images (np.ndarray): Array of images
            
        Returns:
            np.ndarray: Normalized images
        """
        return images / 255.0
    
    def create_augmentation_pipeline(self):
        """
        Create data augmentation pipeline using ImageDataGenerator
        
        Returns:
            ImageDataGenerator: Configured augmentation pipeline
        """
        train_generator = ImageDataGenerator(
            rotation_range=20,           # Random rotation between -20 to 20 degrees
            zoom_range=0.2,              # Random zoom between 0.8 to 1.2
            horizontal_flip=True,        # Random horizontal flip
            brightness_range=[0.8, 1.2], # Brightness adjustment
            fill_mode='nearest'
        )
        
        return train_generator
    
    def augment_batch(self, images, generator, batch_size=32):
        """
        Generate augmented batches from images
        
        Args:
            images (np.ndarray): Input images
            generator (ImageDataGenerator): Augmentation generator
            batch_size (int): Batch size
            
        Returns:
            generator: Batches of augmented images
        """
        return generator.flow(images, batch_size=batch_size, shuffle=False)


def split_dataset(images, labels, test_size=0.2, val_size=0.1):
    """
    Split dataset into training, validation, and test sets
    
    Args:
        images (np.ndarray): Array of images
        labels (np.ndarray): Array of labels
        test_size (float): Proportion of test set (default: 0.2)
        val_size (float): Proportion of validation set (default: 0.1)
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data_for_training(images, labels, test_size=0.2, val_size=0.1):
    """
    Complete data preparation pipeline
    
    Args:
        images (np.ndarray): Raw images
        labels (np.ndarray): Image labels
        test_size (float): Test set proportion
        val_size (float): Validation set proportion
        
    Returns:
        dict: Processed training data
    """
    preprocessor = ImagePreprocessor()
    
    # Normalize images
    images_normalized = preprocessor.normalize_images(images)
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        images_normalized, labels, test_size, val_size
    )
    
    # Convert labels to one-hot encoding
    from tensorflow.keras.utils import to_categorical
    num_classes = len(np.unique(labels))
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train_cat,
        'y_val': y_val_cat,
        'y_test': y_test_cat,
        'y_train_labels': y_train,
        'y_val_labels': y_val,
        'y_test_labels': y_test,
        'num_classes': num_classes
    }
