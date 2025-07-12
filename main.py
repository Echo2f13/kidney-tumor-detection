import os
import shutil
import random
from pathlib import Path

# Set your base dataset directory
SOURCE_DIR = Path("data")
TRAIN_DIR = SOURCE_DIR / "train"
VAL_DIR = SOURCE_DIR / "val"
SPLIT_RATIO = 0.8  # 80% train, 20% val

# Ensure train and val folders are clean
for folder in [TRAIN_DIR, VAL_DIR]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

# Go through each class folder
class_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir() and d.name not in ['train', 'val']]

for class_dir in class_dirs:
    images = list(class_dir.glob("*.*"))  # jpg, png, etc.
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class subfolders
    (TRAIN_DIR / class_dir.name).mkdir(parents=True, exist_ok=True)
    (VAL_DIR / class_dir.name).mkdir(parents=True, exist_ok=True)

    # Copy train images
    for img_path in train_images:
        shutil.copy(img_path, TRAIN_DIR / class_dir.name / img_path.name)

    # Copy val images
    for img_path in val_images:
        shutil.copy(img_path, VAL_DIR / class_dir.name / img_path.name)

    print(f"Processed class '{class_dir.name}': {len(train_images)} train, {len(val_images)} val")

print("\n✅ Dataset split complete.")
