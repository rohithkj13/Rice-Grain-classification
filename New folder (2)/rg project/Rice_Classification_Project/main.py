"""
Main Pipeline for Rice Grain Classification
Orchestrates the complete training and evaluation workflow
Optimized for large datasets using batch loading
"""
from yolo_classifier_pipeline import predict_image
import cv2
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, 'src')

from train_model import TransferLearningModel
from utils.visualization import Visualizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def main():
    """
    Complete training and evaluation pipeline with efficient batch loading
    """
    print("="*70)
    print("RICE GRAIN CLASSIFICATION - COMPLETE PIPELINE")
    print("="*70)
    
    # Configuration
    CONFIG = {
        'dataset_path': 'Rice_Image_Dataset',
        'model_type': 'MobileNetV2',  # 'MobileNetV2' or 'ResNet50'
        'image_size': 224,
        'epochs': 10,
        'batch_size': 32,
        'model_path': 'models/rice_model.h5',
        'use_augmentation': True,
        'fine_tune': True,
        'fine_tune_epochs': 3
    }
    
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Create models and outputs directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Prepare Data Generators
    print("\n" + "-"*70)
    print("STEP 1: PREPARING DATA WITH GENERATORS")
    print("-"*70)
    
    dataset_path = Path(CONFIG['dataset_path'])
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.15
    )
    
    # Load training data
    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        str(dataset_path),
        target_size=(CONFIG['image_size'], CONFIG['image_size']),
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    print(f"Training samples: {train_generator.samples}")
    class_names = list(train_generator.class_indices.keys())
    print(f"Classes: {class_names}")
    
    # Load validation data
    print("\nLoading validation data...")
    val_generator = train_datagen.flow_from_directory(
        str(dataset_path),
        target_size=(CONFIG['image_size'], CONFIG['image_size']),
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"Validation samples: {val_generator.samples}")
    
    # Step 2: Build Model
    print("\n" + "-"*70)
    print("STEP 2: BUILDING TRANSFER LEARNING MODEL")
    print("-"*70)
    
    num_classes = len(class_names)
    
    model = TransferLearningModel(
        num_classes=num_classes,
        model_type=CONFIG['model_type'],
        input_size=CONFIG['image_size']
    )
    
    model.build_model()
    model.get_model_summary()
    
    # Step 3: Train Model
    print("\n" + "-"*70)
    print("STEP 3: TRAINING MODEL")
    print("-"*70)
    
    history = model.model.fit(
        train_generator,
        epochs=CONFIG['epochs'],
        validation_data=val_generator,
        steps_per_epoch=train_generator.samples // CONFIG['batch_size'],
        validation_steps=val_generator.samples // CONFIG['batch_size']
    )
    
    print("\nTraining completed!")
    
    # Step 4: Fine-tune Model (Optional)
    if CONFIG['fine_tune']:
        print("\n" + "-"*70)
        print("STEP 4: FINE-TUNING MODEL")
        print("-"*70)
        
        # Unfreeze last layers
        model.model.layers[0].trainable = True
        
        # Freeze all but last 20 layers
        for layer in model.model.layers[0].layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune
        fine_tune_history = model.model.fit(
            train_generator,
            epochs=CONFIG['fine_tune_epochs'],
            validation_data=val_generator,
            steps_per_epoch=train_generator.samples // CONFIG['batch_size'],
            validation_steps=val_generator.samples // CONFIG['batch_size']
        )
        
        print("\nFine-tuning completed!")
    
    # Step 5: Evaluate Model
    print("\n" + "-"*70)
    print("STEP 5: EVALUATING MODEL")
    print("-"*70)
    
    # Evaluate on validation set
    print("\nEvaluation on Validation Set:")
    val_loss, val_accuracy = model.model.evaluate(val_generator)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Step 6: Save Model
    print("\n" + "-"*70)
    print("STEP 6: SAVING MODEL")
    print("-"*70)
    
    model.save_model(CONFIG['model_path'])
    print(f"Model saved to {CONFIG['model_path']}")
    
    # Step 7: Visualize Results
    print("\n" + "-"*70)
    print("STEP 7: CREATING VISUALIZATIONS")
    print("-"*70)
    
    visualizer = Visualizer()
    
    # Combine training history if fine-tuning
    if CONFIG['fine_tune']:
        combined_history = {
            'loss': history.history['loss'] + fine_tune_history.history['loss'],
            'accuracy': history.history['accuracy'] + fine_tune_history.history['accuracy'],
            'val_loss': history.history['val_loss'] + fine_tune_history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
        }
    else:
        combined_history = history.history
    
    # Plot training history
    print("Plotting training history...")
    visualizer.plot_training_history(
        combined_history,
        save_path='outputs/training_history.png'
    )
    print("✓ Saved: outputs/training_history.png")
    
    # Step 8: Summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults Summary:")
    print(f"  Model Architecture: {CONFIG['model_type']}")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")
    print(f"  Training Samples: {train_generator.samples}")
    print(f"  Validation Samples: {val_generator.samples}")
    print(f"  Training Epochs: {CONFIG['epochs']}")
    if CONFIG['fine_tune']:
        print(f"  Fine-tuning Epochs: {CONFIG['fine_tune_epochs']}")
    print(f"  Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Final Validation Loss: {val_loss:.4f}")
    print(f"\nSaved Files:")
    print(f"  ✓ Model: {CONFIG['model_path']}")
    print(f"  ✓ Visualizations: outputs/")
    
    # Save configuration and metadata
    metadata = {
        'config': CONFIG,
        'class_names': class_names,
        'training_samples': train_generator.samples,
        'validation_samples': val_generator.samples,
        'final_validation_accuracy': float(val_accuracy),
        'final_validation_loss': float(val_loss),
        'num_classes': num_classes
    }
    
    metadata_path = 'outputs/metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata: {metadata_path}")
    
    return model, class_names

def run_inference():
    print("\n" + "="*70)
    print("YOLO + CLASSIFIER INFERENCE MODE")
    print("="*70)

    image_path = input("Enter image path: ")

    final_class, annotated = predict_image(image_path)

    print(f"\nFinal Prediction (Majority): {final_class}")

    # Save output
    output_path = "outputs/result.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved to {output_path}")


if __name__ == '__main__':
    choice = input("Enter 'train' or 'predict': ").strip().lower()

    if choice == 'train':
        main()
    elif choice == 'predict':
        run_inference()
    else:
        print("Invalid option. Use 'train' or 'predict'")

