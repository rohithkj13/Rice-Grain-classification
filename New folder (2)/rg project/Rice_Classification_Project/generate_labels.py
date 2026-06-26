import os

dataset_path = "Rice_Image_Dataset"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(root, file)
            label_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")

            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

print("Labels generated successfully!")