import albumentations as A
import cv2
import os
import random
import glob
from pathlib import Path

# --- CONFIGURATION (Adjust these before running) ---
INPUT_IMGS = "images"  # Folder with source images
INPUT_LBLS = "labels"  # Folder with source labels
OUTPUT_DIR = "dataset/alien_data/class0and1"  # Where to save the 1000 images
TARGET_COUNT = 200  # Total images desired
ALIEN_PROB = 0.95  # 95% chance of Alien colors


# --- THE PIPELINE ---
def get_pipeline():
    return A.Compose([
        # 1. GEOMETRY (Aggressive but Safe)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=40, p=0.7, border_mode=cv2.BORDER_REFLECT),  # +/- 40 deg rotation
        A.Perspective(scale=(0.05, 0.1), p=0.3),  # Slight 3D tilt

        # 2. THE ALIEN LOOK (95% Probability)
        A.OneOf([
            # Radical Color Shifts
            A.HueSaturationValue(hue_shift_limit=100, sat_shift_limit=50, val_shift_limit=40, p=1.0),
            A.ChannelShuffle(p=0.5),
            A.ToGray(p=0.2),
        ], p=ALIEN_PROB),

        # 3. TEXTURE DESTRUCTION (Noise/Blur)
        A.OneOf([
            A.GaussNoise(var_limit=(100.0, 600.0), p=1.0),  # Heavy Grain
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1.0),
            A.Blur(blur_limit=7, p=0.3),
        ], p=ALIEN_PROB),

        # 4. ENVIRONMENT (Lighting)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def process_batch():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_lbl_dir = Path(OUTPUT_DIR) / "labels"  # Create subfolder for labels if you prefer
    # Ideally, YOLO expects images and labels in separate parallel folders.
    # Let's adjust to standard YOLO format:
    out_img_root = Path(OUTPUT_DIR) / "images"
    out_lbl_root = Path(OUTPUT_DIR) / "labels"
    out_img_root.mkdir(parents=True, exist_ok=True)
    out_lbl_root.mkdir(parents=True, exist_ok=True)

    # Load source files
    img_files = glob.glob(os.path.join(INPUT_IMGS, "*.*"))
    img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.png', '.jpeg','.avif','.webp'))]

    if not img_files:
        print("❌ No images found!")
        return

    print(f"🏭 Factory started. Source: {len(img_files)} images. Target: {TARGET_COUNT}.")

    transform = get_pipeline()
    generated = 0

    while generated < TARGET_COUNT:
        # Pick random source image
        img_path = random.choice(img_files)
        lbl_path = os.path.join(INPUT_LBLS, Path(img_path).stem + ".txt")

        # Read Image
        image = cv2.imread(img_path)
        if image is None: continue
        h, w, _ = image.shape

        # Read Label (YOLO format)
        bboxes = []
        class_labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    # Albumentations requires normalized YOLO coords directly
                    bboxes.append(coords)
                    class_labels.append(cls)

        # Apply Transformation
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_cls = augmented['class_labels']

            # Save Image
            fname = f"alien_{generated:05d}.jpg"
            cv2.imwrite(str(out_img_root / fname), aug_img)

            # Save Label
            with open(out_lbl_root / f"alien_{generated:05d}.txt", "w") as f:
                for cls, box in zip(aug_cls, aug_bboxes):
                    # Clamp values to 0-1 just in case rotation pushed them slightly out
                    box = [min(max(x, 0.0), 1.0) for x in box]
                    f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

            generated += 1
            if generated % 100 == 0:
                print(f"   ⚡ Generated {generated}/{TARGET_COUNT}...")

        except Exception as e:
            # Augmentation can sometimes fail with aggressive geometric transforms on edge cases
            continue

    print(f"✅ Mission Complete. Data generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    process_batch()