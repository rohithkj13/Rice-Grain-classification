"""
Model Training Module for Rice Classification
Implements stronger transfer learning with MobileNetV2, EfficientNetB0, or ResNet50.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.optimizers import Adam
from preprocess import ImagePreprocessor, prepare_data_for_training
import json


class TransferLearningModel:
    """
    Transfer Learning model builder using pretrained CNNs
    """
    
    def __init__(self, num_classes=5, model_type='MobileNetV2', input_size=224):
        """
        Initialize transfer learning model
        
        Args:
            num_classes (int): Number of rice classes (default: 5)
            model_type (str): 'MobileNetV2' or 'ResNet50' (default: MobileNetV2)
            input_size (int): Input image size (default: 224)
        """
        self.num_classes = num_classes
        self.model_type = model_type
        self.input_size = input_size
        self.model = None
        self.history = None
        self.base_model = None
        self.preprocess_fn = None
        self.callbacks = []
        
    def build_model(self):
        """
        Build transfer learning model with custom classifier
        
        Returns:
            keras.Sequential: Compiled model
        """
        print(f"\nBuilding {self.model_type} Transfer Learning Model...")
        
        # Load pretrained model without top layers
        if self.model_type == 'MobileNetV2':
            base_model = MobileNetV2(
                input_shape=(self.input_size, self.input_size, 3),
                include_top=False,
                weights='imagenet'
            )
            self.preprocess_fn = mobilenet_preprocess
        elif self.model_type == 'EfficientNetB0':
            base_model = EfficientNetB0(
                input_shape=(self.input_size, self.input_size, 3),
                include_top=False,
                weights='imagenet'
            )
            self.preprocess_fn = efficientnet_preprocess
        elif self.model_type == 'ResNet50':
            base_model = ResNet50(
                input_shape=(self.input_size, self.input_size, 3),
                include_top=False,
                weights='imagenet'
            )
            self.preprocess_fn = resnet_preprocess
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Freeze base model layers
        base_model.trainable = False
        self.base_model = base_model
        print(f"Base model layers frozen: {len(base_model.layers)} layers")
        
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.1),
        ], name='augmentation')

        inputs = keras.Input(shape=(self.input_size, self.input_size, 3), name='image')
        x = augmentation(inputs)
        x = layers.Lambda(self.preprocess_fn, name='preprocess')(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.35, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.25, name='dropout_2')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name=f'{self.model_type}_rice_classifier')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
        )
        
        self.callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=6,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
        ]
        self.model = model
        print("Model built and compiled successfully!")
        
        return model
    
    def get_model_summary(self):
        """
        Print model summary
        """
        if self.model is None:
            print("Model not built yet!")
            return
        
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=15, batch_size=32, augmentation=False):
        """
        Train the model
        
        Args:
            X_train (np.ndarray): Training images
            y_train (np.ndarray): Training labels (one-hot encoded)
            X_val (np.ndarray): Validation images
            y_val (np.ndarray): Validation labels
            epochs (int): Number of training epochs (default: 15)
            batch_size (int): Batch size (default: 32)
            augmentation (bool): Apply data augmentation (default: False)
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            self.build_model()
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def fine_tune(self, X_train, y_train, X_val, y_val, 
                  epochs=5, batch_size=32, unfreeze_layers=50):
        """
        Fine-tune the base model by unfreezing some layers
        
        Args:
            X_train (np.ndarray): Training images
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation images
            y_val (np.ndarray): Validation labels
            epochs (int): Number of fine-tuning epochs
            batch_size (int): Batch size
            unfreeze_layers (int): Number of layers to unfreeze from the end
        """
        if self.model is None:
            raise ValueError("Model not built yet!")
        
        print(f"\nFine-tuning: unfreezing last {unfreeze_layers} layers...")
        
        # Get base model
        base_model = self.base_model
        
        # Unfreeze layers
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
        )
        
        # Train
        fine_tune_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=1
        )
        
        print("Fine-tuning completed!")
        return fine_tune_history
    
    def save_model(self, save_path='models/rice_model.h5'):
        """
        Save trained model to disk
        
        Args:
            save_path (str): Path to save the model
        """
        if self.model is None:
            print("No model to save!")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path='models/rice_model.h5'):
        """
        Load trained model from disk
        
        Args:
            load_path (str): Path to load the model
        """
        self.model = keras.models.load_model(load_path)
        print(f"Model loaded from {load_path}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test (np.ndarray): Test images
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            print("Model not loaded!")
            return None
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nModel Evaluation:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def predict_batch(self, images):
        """
        Make predictions on a batch of images
        
        Args:
            images (np.ndarray): Batch of normalized images
            
        Returns:
            np.ndarray: Predictions (class probabilities)
        """
        if self.model is None:
            print("Model not loaded!")
            return None
        
        return self.model.predict(images)
    
    def extract_features(self, images):
        """
        Extract features from second-to-last layer for KNN
        
        Args:
            images (np.ndarray): Input images
            
        Returns:
            np.ndarray: Feature vectors
        """
        if self.model is None:
            print("Model not loaded!")
            return None
        
        # Create feature extraction model (up to dense_2 layer)
        feature_model = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('dense_2').output
        )
        
        features = feature_model.predict(images, verbose=0)
        return features
