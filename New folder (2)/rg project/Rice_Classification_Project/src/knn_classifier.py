"""
KNN Classifier Module for Rice Classification
Uses CNN extracted features to train KNN classifier
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os


class KNNClassifier:
    """
    K-Nearest Neighbors classifier using CNN features
    """
    
    def __init__(self, n_neighbors=5):
        """
        Initialize KNN classifier
        
        Args:
            n_neighbors (int): Number of neighbors to consider (default: 5)
        """
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False
    
    def train(self, features, labels):
        """
        Train KNN classifier on extracted features
        
        Args:
            features (np.ndarray): CNN extracted features
            labels (np.ndarray): Image labels (integers)
        """
        print(f"\nTraining KNN classifier with {self.n_neighbors} neighbors...")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Number of training samples: {len(features)}")
        
        self.classifier.fit(features, labels)
        self.is_trained = True
        
        print("KNN training completed!")
    
    def predict(self, features):
        """
        Make predictions on feature vectors
        
        Args:
            features (np.ndarray): CNN extracted features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        
        predictions = self.classifier.predict(features)
        return predictions
    
    def predict_with_confidence(self, features):
        """
        Make predictions with confidence scores
        
        Args:
            features (np.ndarray): CNN extracted features
            
        Returns:
            tuple: (predictions, distances, confidences)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        
        predictions = self.classifier.predict(features)
        distances, indices = self.classifier.kneighbors(features)
        
        # Calculate confidence as inverse of average distance
        confidences = 1.0 / (1.0 + np.mean(distances, axis=1))
        
        return predictions, distances, confidences
    
    def evaluate(self, features, labels):
        """
        Evaluate KNN performance
        
        Args:
            features (np.ndarray): CNN extracted features
            labels (np.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        
        accuracy = self.classifier.score(features, labels)
        return accuracy
    
    def save_classifier(self, save_path='models/knn_classifier.pkl'):
        """
        Save trained classifier to disk
        
        Args:
            save_path (str): Path to save the classifier
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        print(f"KNN classifier saved to {save_path}")
    
    def load_classifier(self, load_path='models/knn_classifier.pkl'):
        """
        Load trained classifier from disk
        
        Args:
            load_path (str): Path to load the classifier
        """
        with open(load_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        self.is_trained = True
        print(f"KNN classifier loaded from {load_path}")
    
    def get_neighbor_distances(self, features, n_samples=5):
        """
        Get nearest neighbor distances for debugging
        
        Args:
            features (np.ndarray): Input features
            n_samples (int): Number of samples to show
            
        Returns:
            dict: Neighbor information
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        
        distances, indices = self.classifier.kneighbors(features[:n_samples])
        
        neighbor_info = {
            'distances': distances.tolist(),
            'indices': indices.tolist()
        }
        
        return neighbor_info
