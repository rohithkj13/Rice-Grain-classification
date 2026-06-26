import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = r"C:/Users/ROHITH/Downloads/New folder (2) (3)/New folder (2)/rg project/Rice_Classification_Project/Rice_Image_Dataset_Reduced"
MODEL_SAVE_PATH = "models/rice_model_advanced.h5"
CLASSES_CONFIG_PATH = "models/classes.json"
OUTPUTS_DIR = "outputs"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

def train():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        return
        
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Setup Data Generators mapping exactly to training
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False # Shuffle False is important for Confusion Matrix logic later
    )

    # 2. Save Classes Mapping
    class_indices = train_generator.class_indices
    # Invert mapping to {"0": "Arborio", "1": "Basmati", ...} NOTE: JSON requires string keys
    index_to_class = {str(v): k for k, v in class_indices.items()}
    with open(CLASSES_CONFIG_PATH, "w") as f:
        json.dump(index_to_class, f, indent=4)
    print(f"Saved class mapping safely to {CLASSES_CONFIG_PATH}: {index_to_class}")

    # 3. Handle Class Imbalance Weighting
    classes_train = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes_train),
        y=classes_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Calculated Class Weights to Fix Imbalance: {class_weight_dict}")

    # 4. Model Architecture (Linear Activation for Temperature Scaling later)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base layers first
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)  # Dropout to prevent overfitting
    
    # CRITICAL: We want logits so we can use Temperature Scaling in prediction!
    # activation=None output means these are raw logits.
    predictions = Dense(len(class_indices), activation=None)(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("Phase 1: Training top layers...")
    model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

    # 5. Fine-Tuning 
    print("Phase 2: Unfreezing top 20 layers for fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        epochs=6,
        validation_data=val_generator,
        class_weight=class_weight_dict
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Model successfully trained and saved to: {MODEL_SAVE_PATH}")
    
    # 6. Evaluation metrics mapping
    print("Generating Evaluation Metrics & Confusion Matrix...")
    y_pred_logits = model.predict(val_generator)
    y_pred = np.argmax(y_pred_logits, axis=1)
    y_true = val_generator.classes
    
    class_names = [index_to_class[str(i)] for i in range(len(class_indices))]
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Validation Confusion Matrix')
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved to {cm_path}")
    
    # Save Classification Report
    cr = classification_report(y_true, y_pred, target_names=class_names)
    cr_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(cr_path, 'w') as f:
        f.write(cr)
    print(f"Classification Report saved to {cr_path}")
    print("\nTraining Pipeline Completed Successfully.")

if __name__ == "__main__":
    train()
