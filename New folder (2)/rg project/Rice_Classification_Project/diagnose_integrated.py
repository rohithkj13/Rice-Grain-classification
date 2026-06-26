"""Diagnostic test to debug pipeline behavior"""

import os
import cv2
from src.advanced_integrated_classifier import AdvancedIntegratedClassifier

classifier = AdvancedIntegratedClassifier()
dataset_path = 'Rice_Image_Dataset'

# Test on a few specific images
test_cases = [
    ('Basmati', 'Basmati (1).jpg'),
    ('Jasmine', 'Jasmine (1).jpg'),
    ('Arborio', 'Arborio (1).jpg'),
]

print("\n" + "="*80)
print("DIAGNOSTIC: Advanced Integrated Classifier Debug")
print("="*80)

for rice_type, filename in test_cases:
    img_path = os.path.join(dataset_path, rice_type, filename)
    
    if not os.path.exists(img_path):
        print(f"\n❌ Image not found: {img_path}")
        continue
    
    print(f"\n{'='*80}")
    print(f"Image: {rice_type}/{filename}")
    print('='*80)
    
    # Load image
    image = cv2.imread(img_path)
    
    # Check image type
    img_type = classifier._detect_image_type(image)
    print(f"Detected type: {img_type}")
    
    # Get features
    if img_type == 'single':
        contour = classifier._extract_grain_contour(image)
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            ar = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            print(f"Aspect ratio: {ar:.3f}")
            
            # Check shape rule
            shape_pred = classifier._apply_shape_rule(ar)
            print(f"Shape rule prediction: {shape_pred}")
    
    # Run full classification
    result = classifier.classify(img_path)
    
    print(f"\nFinal Result:")
    print(f"  Type: {result.get('finalType')}")
    print(f"  Confidence: {result.get('confidence'):.3f}")
    print(f"  Source: {result.get('source')}")
    print(f"  Stable: {result.get('stable')}")
    
    # Show step details
    if result.get('details', {}).get('candidates'):
        print(f"\nCandidates:")
        for cand in result['details']['candidates']:
            print(f"  - {cand['method']}: {cand['type']} (conf: {cand['confidence']:.3f}, priority: {cand['priority']})")
    else:
        print(f"\nNo candidates generated!")
        if result.get('details', {}).get('debug'):
            print("Debug info:")
            for debug in result['details']['debug']:
                print(f"  {debug}")
    
    # Check if correct
    actual = rice_type
    predicted = result.get('finalType')
    status = "✓ CORRECT" if predicted == actual else "❌ WRONG"
    print(f"\nActual: {actual}, Predicted: {predicted} {status}")
