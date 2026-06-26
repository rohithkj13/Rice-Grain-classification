"""
Visualization Module for Rice Classification
Plots training metrics, confusion matrix, and other visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Visualizer:
    """
    Visualization utilities for rice classification project
    """
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        Plot training accuracy and loss over epochs
        
        Args:
            history (dict): Training history from model.fit()
            save_path (str): Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0].plot(history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[1].plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
        """
        Plot confusion matrix as heatmap
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (list): List of class names
            save_path (str): Path to save figure (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_class_distribution(labels, class_names, save_path=None):
        """
        Plot distribution of samples across classes
        
        Args:
            labels (np.ndarray): Array of class labels
            class_names (list): List of class names
            save_path (str): Path to save figure (optional)
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([class_names[i] for i in unique], counts, color='steelblue', edgecolor='black')
        plt.xlabel('Rice Variety', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
        plt.title('Distribution of Rice Grain Samples', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(cnn_metrics, knn_metrics, save_path=None):
        """
        Compare CNN vs KNN performance
        
        Args:
            cnn_metrics (dict): CNN metrics
            knn_metrics (dict): KNN metrics
            save_path (str): Path to save figure (optional)
        """
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        cnn_values = [
            cnn_metrics['accuracy'],
            cnn_metrics['precision_macro'],
            cnn_metrics['recall_macro'],
            cnn_metrics['f1_macro']
        ]
        knn_values = [
            knn_metrics['accuracy'],
            knn_metrics['precision_macro'],
            knn_metrics['recall_macro'],
            knn_metrics['f1_macro']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, cnn_values, width, label='CNN', color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, knn_values, width, label='KNN', color='coral', edgecolor='black')
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('CNN vs KNN Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_sample_predictions(predictor, images, labels, class_names, num_samples=6, save_path=None):
        """
        Plot sample images with predictions
        
        Args:
            predictor: RicePredictor instance
            images (np.ndarray): Sample images
            labels (np.ndarray): True labels
            class_names (list): List of class names
            num_samples (int): Number of samples to plot
            save_path (str): Path to save figure (optional)
        """
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        predictions = predictor.model.predict(images[:num_samples], verbose=0)
        
        for i in range(num_samples):
            ax = axes[i]
            
            # Display image
            image_display = (images[i] * 255).astype(np.uint8)
            ax.imshow(image_display)
            
            # Get prediction info
            pred_class_idx = np.argmax(predictions[i])
            true_class_idx = labels[i]
            confidence = predictions[i][pred_class_idx]
            
            true_class = class_names[true_class_idx]
            pred_class = class_names[pred_class_idx]
            
            # Color based on correctness
            correct = pred_class_idx == true_class_idx
            title_color = 'green' if correct else 'red'
            
            title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2%}"
            ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)
            ax.axis('off')
        
        # Hide extra subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_per_class_metrics(class_names, precision, recall, f1, save_path=None):
        """
        Plot per-class performance metrics
        
        Args:
            class_names (list): List of class names
            precision (np.ndarray): Precision scores per class
            recall (np.ndarray): Recall scores per class
            f1 (np.ndarray): F1 scores per class
            save_path (str): Path to save figure (optional)
        """
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', color='steelblue', edgecolor='black')
        ax.bar(x, recall, width, label='Recall', color='coral', edgecolor='black')
        ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('Rice Variety', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")
        
        plt.show()
