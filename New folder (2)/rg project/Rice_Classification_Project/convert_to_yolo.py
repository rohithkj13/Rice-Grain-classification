import os
import shutil

source = "Rice_Image_Dataset_Reduced"
dest_images = "dataset/images"
dest_labels = "dataset/labels"

# 🔥 DELETE OLD DATASET (VERY IMPORTANT)
if os.path.exists("dataset"):
    shutil.rmtree("dataset")

os.makedirs(dest_images)
os.makedirs(dest_labels)

count = 0

for class_name in os.listdir(source):
    class_path = os.path.join(source, class_name)

    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        if img_name.endswith((".jpg", ".png", ".jpeg")):
            src_img = os.path.join(class_path, img_name)

            new_name = f"{class_name}_{count}.jpg"
            dst_img = os.path.join(dest_images, new_name)
            dst_label = os.path.join(dest_labels, new_name.replace(".jpg", ".txt"))

            shutil.copy(src_img, dst_img)

            with open(dst_label, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

            count += 1

print("✅ Dataset converted to YOLO format!")