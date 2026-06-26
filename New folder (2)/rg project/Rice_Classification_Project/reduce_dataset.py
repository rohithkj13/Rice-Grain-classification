import os
import shutil
import random

SOURCE_DIR = r"C:/Users/ROHITH/Downloads/New folder (2) (3)/New folder (2)/rg project/Rice_Classification_Project/Rice_Image_Dataset"

TARGET_DIR = r"C:/Users/ROHITH/Downloads/New folder (2) (3)/New folder (2)/rg project/Rice_Classification_Project/Rice_Image_Dataset_Reduced"

IMAGES_PER_CLASS = 1500  # change (3000–5000 recommended)

os.makedirs(target_class_path, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    target_class_path = os.path.join(TARGET_DIR, class_name)
    os.makedirs(target_class_path, exist_ok=True)

    images = [img for img in os.listdir(class_path)
              if img.endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(images)

    selected_images = images[:IMAGES_PER_CLASS]

    for img in selected_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(target_class_path, img)

        shutil.copy(src, dst)

    print(f"✅ {class_name}: {len(selected_images)} images copied")

print("\n🔥 Dataset reduction completed!")