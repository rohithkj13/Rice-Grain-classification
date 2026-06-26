import os

def setup_non_rice_directory():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'Rice_Image_Dataset')
    non_rice_dir = os.path.join(dataset_dir, 'Non-Rice')
    
    if not os.path.exists(non_rice_dir):
        os.makedirs(non_rice_dir)
        print(f"Created directory: {non_rice_dir}")
    
    readme_path = os.path.join(non_rice_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("""CRITICAL ACTION REQUIRED: Populating 'Non-Rice' Class

Your model needs to learn what IS NOT rice!
To prevent the model from blindly guessing a rice class when given a non-rice image, you MUST provide real-world examples of non-rice images.

Instructions:
1. Download 300 to 500 images of the following items:
   - Wheat
   - Dal / Pulses
   - Small Stones / Gravel
   - Soil / Sand
   - Random kitchen table backgrounds
   - Other random objects
2. Place all these images (jpg, png) directly inside this folder ("Non-Rice").
3. DO NOT use plain white noise images or solid colors. Use real images.
4. Once you have at least 300 images here, you can run `model_training.py`!
""")

    print("\n" + "="*60)
    print("ACTION REQUIRED: Please populate the 'Non-Rice' directory")
    print("="*60)
    print(f"Directory located at: {non_rice_dir}")
    print("Read the README.txt inside that folder for instructions.")
    print("You must add 300-500 real images of wheat, dal, stones, etc.,")
    print("before running model_training.py to avoid 'noise' overfitting.\n")

if __name__ == '__main__':
    setup_non_rice_directory()
