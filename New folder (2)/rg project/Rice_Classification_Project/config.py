"""
Configuration and settings for Rice Classification Project
"""

# Model settings
MODEL_CONFIG = {
    'input_size': 224,
    'num_classes': 5,
    'class_names': ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'],
    'model_type': 'MobileNetV2'  # or 'ResNet50'
}

# Training settings
TRAINING_CONFIG = {
    'epochs': 15,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'fine_tune_learning_rate': 1e-5,
    'fine_tune_epochs': 5,
    'validation_split': 0.15,
    'test_split': 0.2,
    'use_augmentation': True
}

# Data augmentation settings
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# Paths
PATHS = {
    'dataset': 'dataset',
    'models_dir': 'models',
    'outputs_dir': 'outputs',
    'model_file': 'models/rice_model.h5',
    'knn_file': 'models/knn_classifier.pkl'
}

# KNN settings
KNN_CONFIG = {
    'n_neighbors': 5,
    'algorithm': 'auto'
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figsize': (14, 5),
    'dpi': 300,
    'style': 'seaborn',
    'colormap': 'viridis'
}
