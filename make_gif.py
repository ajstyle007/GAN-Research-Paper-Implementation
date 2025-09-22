# import imageio
# import glob

# # Collect your images (make sure they are sorted)
# images = []
# for filename in sorted(glob.glob("images/*.png")):  # adjust folder/extension
#     images.append(imageio.imread(filename))

# # Save as GIF
# imageio.mimsave("training_progress2.gif", images, duration=10)  
# # duration=0.5 → 0.5s per frame

import imageio
import glob

# Use glob to find all PNGs in samples2 folder
image_paths = sorted(glob.glob("samples2/*.png"))  # adjust folder path if needed

# Read images
images = [imageio.v2.imread(path) for path in image_paths]

# Make GIF slower by repeating frames
images_slow = []
for img in images:
    images_slow.extend([img]*10)  # repeat each frame 10 times

imageio.mimsave("training_progress3.gif", images_slow, duration=0.2)
