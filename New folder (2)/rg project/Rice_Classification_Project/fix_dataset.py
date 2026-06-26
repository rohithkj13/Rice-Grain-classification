import os
import shutil
import random

SOURCE_DIR = "Rice_Image_Dataset_Reduced"
TARGET_DIR = "Rice_Image_Dataset_Final"

IMAGES_PER_CLASS = 1500

# Delete target if exists
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

os.makedirs(TARGET_DIR)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    target_class_path = os.path.join(TARGET_DIR, class_name)
    os.makedirs(target_class_path)

    images = [img for img in os.listdir(class_path)
              if img.endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(images)
    selected_images = images[:IMAGES_PER_CLASS]

    for img in selected_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(target_class_path, img)

        shutil.copy(src, dst)

    print(f"✅ {class_name}: {len(selected_images)} images copied")

print("🔥 Final dataset ready!")