import albumentations as A
import cv2
import os
import glob
import numpy as np

# --- CONFIGURATION ---
INPUT_IMAGES_DIR = "images"
INPUT_LABELS_DIR = "labels"
OUTPUT_DIR = "v25_finetune_dataset"  # Output folder

# Class Map (0: mine, 1: rov, 2: torpedo, 3: explosive)
CLASS_MAPPING = [0, 1, 2, 3]


def get_combinatorial_pipelines():
    """
    Defines 5 UNIQUE Combinations using EXPLICIT TUPLES for ranges.
    This fixes the 'min <= max' validation error.
    """
    return {
        # 1. MIRRORED ALGAE (Horizontal Flip + Green + Blur)
        "1_MirroredAlgae": A.Compose([
            A.HorizontalFlip(p=1.0),
            A.SafeRotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0),

            # FIXED: Explicit tuples (min, max) for every shift
            A.RGBShift(
                r_shift_limit=(-0.25, -0.15),  # Remove Red
                g_shift_limit=(0.15, 0.25),  # Add Green
                b_shift_limit=(-0.25, -0.15),  # Remove Blue
                p=1.0
            ),
            A.GaussianBlur(blur_limit=(7, 11), p=1.0),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

        # 2. INVERTED ABYSS (Vertical Flip + Blue + Dark + Noise)
        "2_InvertedAbyss": A.Compose([
            A.VerticalFlip(p=1.0),
            A.SafeRotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=1.0),

            A.ToGray(p=0.2),
            A.RGBShift(
                r_shift_limit=(-0.3, -0.2),  # Remove Red
                g_shift_limit=(-0.05, 0.05),  # Neutral Green
                b_shift_limit=(0.2, 0.3),  # Add Blue
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, -0.2),  # Darker
                contrast_limit=(0.1, 0.3),  # More contrast
                p=1.0
            ),
            A.GaussNoise(var_limit=(0.01, 0.03), p=1.0),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

        # 3. DOUBLE FLIP RUST (H-Flip + V-Flip + Red/Brown + Sharpen)
        "3_DoubleFlipRust": A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),

            A.RGBShift(
                r_shift_limit=(0.15, 0.25),  # Add Red (Rust)
                g_shift_limit=(-0.15, -0.05),  # Remove Green
                b_shift_limit=(-0.25, -0.15),  # Remove Blue
                p=1.0
            ),
            A.Sharpen(alpha=(0.5, 1.0), lightness=(1.0, 1.5), p=1.0),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

        # 4. HARD TILT TURBULENCE (Max Rotation + Distortion)
        "4_HardTiltTurbulence": A.Compose([
            A.SafeRotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=1.0),

            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
            A.MotionBlur(blur_limit=(9, 15), p=1.0),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

        # 5. DIAGONAL CHAOS (H-Flip + Rotate + Elastic)
        "5_DiagonalChaos": A.Compose([
            A.HorizontalFlip(p=1.0),
            A.SafeRotate(limit=(15, 20), border_mode=cv2.BORDER_REFLECT_101, p=1.0),

            A.ElasticTransform(
                alpha=100,
                sigma=10,
                alpha_affine=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
    }


def read_yolo_labels(label_path):
    boxes = []
    classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    boxes.append([x, y, w, h])
                    classes.append(int(cls))
    return boxes, classes


def save_yolo_labels(boxes, classes, output_path):
    with open(output_path, 'w') as f:
        for box, cls in zip(boxes, classes):
            x, y, w, h = box
            # SAFETY CLIP: Ensure floating point errors don't push box to 1.000001 or < 0
            x = max(0.0001, min(0.9999, x))
            y = max(0.0001, min(0.9999, y))
            w = max(0.0001, min(0.9999, w))
            h = max(0.0001, min(0.9999, h))

            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    images_out = os.path.join(OUTPUT_DIR, "images")
    labels_out = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    pipelines = get_combinatorial_pipelines()
    image_files = glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.png"))

    print(f"Found {len(image_files)} seeds. Starting COMBINATORIAL augmentation...")

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(INPUT_LABELS_DIR, base_name + ".txt")

        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Albumentations uses RGB

        boxes, classes = read_yolo_labels(label_path)
        if not boxes:
            print(f"Skipping {base_name} - No Labels")
            continue

        # GENERATE 5 VARIANTS FOR THIS SEED
        for name, aug in pipelines.items():
            try:
                # Apply Augmentation
                augmented = aug(image=image, bboxes=boxes, class_labels=classes)

                # Verify we didn't lose the object
                if len(augmented['bboxes']) > 0:
                    out_name = f"{base_name}_{name}"

                    # Save Image (Convert back to BGR)
                    cv2.imwrite(os.path.join(images_out, out_name + ".jpg"),
                                cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))

                    # Save Label
                    save_yolo_labels(augmented['bboxes'], augmented['class_labels'],
                                     os.path.join(labels_out, out_name + ".txt"))
                    print(f"  + Generated: {out_name}")
                else:
                    print(f"  ! Warning: {name} pushed object out of bounds (Skipped)")

            except Exception as e:
                print(f"  ! CRASH on {base_name} / {name}: {e}")

    print("\n------------------------------------------------")
    print(f"COMPLETE. Created distinct dataset at: {OUTPUT_DIR}")
    print("------------------------------------------------")


if __name__ == "__main__":
    main()