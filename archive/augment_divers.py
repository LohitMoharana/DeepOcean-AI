import cv2
import numpy as np
import os
import random
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = "negatives"  # Where your 77 diver images are
OUTPUT_IMG_DIR = "datasets_ft2/negatives/aug_images"  # New folder for augmented versions
OUTPUT_LBL_DIR = "datasets_ft2/negatives/aug_labels"  # New folder for empty labels

# How many augmented versions to create per original image?
# 77 images * 4 variations = 308 new images
VARIATIONS_PER_IMAGE = 4


def add_noise(image):
    """Adds Gaussian noise to simulate underwater grain/turbidity."""
    row, col, ch = image.shape
    mean = 0
    # Sigma controls noise amount. Higher = grainier.
    sigma = random.randint(15, 35)

    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)

    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def shift_hue(image):
    """Radically shifts hue to simulate Green/Brown/Alien water."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)

    # Shift Hue (channel 0) by a random amount (0-179)
    shift = random.randint(20, 160)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

    # Randomly mess with Saturation/Value to simulate murky vs clear
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.5, 1.2), 0, 255)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 1.0), 0, 255)  # Brightness

    img_hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def process_images():
    # Setup directories
    img_path = Path(INPUT_DIR)
    out_img_path = Path(OUTPUT_IMG_DIR)
    out_lbl_path = Path(OUTPUT_LBL_DIR)

    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_lbl_path, exist_ok=True)

    files = [f for f in img_path.iterdir() if f.suffix.lower() in {'.jpg', '.png', '.jpeg','.webp','.avif'}]

    if not files:
        print(f"❌ No images found in {INPUT_DIR}")
        return

    print(f"🚀 Starting augmentation on {len(files)} images...")

    total_generated = 0

    for file in files:
        original = cv2.imread(str(file))
        if original is None: continue

        # Create X variations for each image
        for i in range(VARIATIONS_PER_IMAGE):
            # 1. Apply Effects
            aug_img = shift_hue(original)  # Change color first
            aug_img = add_noise(aug_img)  # Add grain last

            # 2. Save Image
            new_filename = f"{file.stem}_aug_{i}.jpg"
            save_path = out_img_path / new_filename
            cv2.imwrite(str(save_path), aug_img)

            # 3. Create Empty Label (Crucial for Negatives)
            label_filename = f"{file.stem}_aug_{i}.txt"
            with open(out_lbl_path / label_filename, "w") as f:
                pass  # Empty file = "Nothing to see here"

            total_generated += 1

    print("-" * 40)
    print(f"✅ Done! Generated {total_generated} augmented images.")
    print(f"📂 Images: {OUTPUT_IMG_DIR}")
    print(f"📂 Labels: {OUTPUT_LBL_DIR}")
    print("-" * 40)


if __name__ == "__main__":
    process_images()