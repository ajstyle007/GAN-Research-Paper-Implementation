import os
import shutil
import random

# Source and destination folders
src_folder = "img_align_celeba/img_align_celeba"
dst_folder = "test"

# Make sure destination exists
os.makedirs(dst_folder, exist_ok=True)

# Get all images from source folder
all_images = os.listdir(src_folder)

# Randomly pick 20,000 images
selected_images = random.sample(all_images, 20000)

# Copy them
for img in selected_images:
    src_path = os.path.join(src_folder, img)
    dst_path = os.path.join(dst_folder, img)

    shutil.copy(src_path, dst_path)   # for copy
    # shutil.move(src_path, dst_path)  # for move (cut-paste)

print("✅ Done! 20,000 images copied.")
