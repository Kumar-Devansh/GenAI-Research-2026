"""
STEP 8A: SYNTHETIC SATELLITE IMAGE GENERATOR
============================================

IMPORTANT NOTE: Real satellite imagery (Sentinel-2, MODIS, Landsat)
requires Google Earth Engine authentication and large downloads,
which can be impractical for a quick learning demo.

Therefore, this script generates REALISTIC-LOOKING synthetic
"sky/cloud" images in three categories:

1. clear_sky      - clear sky (few clouds)
2. partly_cloudy  - some clouds
3. heavy_clouds   - many clouds (rain-likely conditions)

These images are designed to look similar to real satellite tiles
(with texture/noise patterns) so that the CNN training pipeline can
work with real data as well. You only need to replace this script
with real Sentinel-2 image download code.
"""


import numpy as np
import cv2
import os

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "..", "data", "sky_images")

CATEGORIES = ["clear_sky", "partly_cloudy", "heavy_clouds"]
IMAGES_PER_CATEGORY = 150
IMG_SIZE = 64  # 64x64 pixels - kept small for faster training on CPU


for category in CATEGORIES:
    os.makedirs(os.path.join(IMG_DIR, category), exist_ok=True)

def generate_cloud_image(cloud_density):
   """
 cloud_density: from 0.0 (clear sky) to 1.0 (heavy clouds)
 Generates a Perlin-noise-like pattern using multiple
 layers of smoothed random noise to create a realistic
 cloud-like texture.
"""


# Base sky color (blue gradient - similar to a clear sky)

    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    sky_blue = np.array([235, 206, 135])  # BGR format (OpenCV uses BGR)
    img[:, :] = sky_blue

  # Add multi-scale noise layers to create a cloud-like texture
    cloud_layer = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for scale in [4, 8, 16, 32]:
        small = np.random.rand(scale, scale).astype(np.float32)
        upscaled = cv2.resize(small, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        cloud_layer += upscaled / scale

    cloud_layer = (cloud_layer - cloud_layer.min()) / (cloud_layer.max() - cloud_layer.min())

    # Cloud density threshold - higher density means more pixels will be "cloudy"
    cloud_mask = (cloud_layer > (1 - cloud_density)).astype(np.float32)
    cloud_mask = cv2.GaussianBlur(cloud_mask, (5, 5), 0)

    white = np.array([250, 250, 250])
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - cloud_mask * 0.85) + white[c] * (cloud_mask * 0.85)

# Add a small amount of random noise (similar to sensor noise in realistic satellite images)
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ---- IMAGES GENERATE KARO ----
density_ranges = {
    "clear_sky": (0.0, 0.2),
    "partly_cloudy": (0.35, 0.6),
    "heavy_clouds": (0.7, 0.95),
}

total_generated = 0
for category, (low, high) in density_ranges.items():
    for i in range(IMAGES_PER_CATEGORY):
        density = np.random.uniform(low, high)
        img = generate_cloud_image(density)
        filepath = os.path.join(IMG_DIR, category, f"{category}_{i:03d}.png")
        cv2.imwrite(filepath, img)
        total_generated += 1

print(f"Total {total_generated} synthetic sky images generated in: {IMG_DIR}")
print(f"Categories: {CATEGORIES}, {IMAGES_PER_CATEGORY} images each")
