import albumentations as A
import cv2
import os
import glob
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# 1. Processed Synthetic Bulk
SYNTHETIC_DIR = r"D:\Projects\Personal\DeepOcean_V2.0\training\DeepOcean_R26\dataset_R26_clean"

# 2. Real-World Curated Patch Data
PATCH_POS_DIR = r"D:\Projects\Personal\DeepOcean_V2.0\training\DeepOcean_R26\raw_patch_data\positives/images"
PATCH_NEG_DIR = r"D:\Projects\Personal\DeepOcean_V2.0\training\DeepOcean_R26\raw_patch_data\negatives/images"

# 3. Output Location
OUTPUT_DIR = r"D:\Projects\Personal\DeepOcean_V2.0\training\DeepOcean_R26\r26_colorblind_finetune_dataset"

# 4. STRICT QUOTA BUDGET
TARGET_PER_CLASS = 800  # 800 images per class (0: sea_mine, 1: uuv, 2: diver, 3: misc)
TARGET_NEGATIVES = 1000  # 1000 hard negatives / empty backgrounds
SPLIT_RATIO = 0.8  # 80% Train, 20% Val

CLASSES = ['sea_mine', 'uuv', 'diver', 'misc']


def get_colorblind_pipeline():
    """ Colorblind augmentation pipeline applied during train split compilation """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.SafeRotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.OneOf([
            A.ToGray(p=1.0),
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=0, p=1.0),
            A.RGBShift(r_shift_limit=(-0.25, -0.15), g_shift_limit=(0.15, 0.25), b_shift_limit=(-0.25, -0.15), p=1.0),
        ], p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_and_clamp_labels(label_path):
    """ Bulletproof YOLO label reader. Handles floats, extra columns, and clamping. """
    boxes, classes = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()

                # Accept 5 columns (Standard YOLO) OR 6 columns (Auto-annotated with confidence)
                if len(parts) >= 5:
                    # Force float strings like "2.0" into proper integer 2
                    cls = int(float(parts[0]))

                    # IF your divers were accidentally exported as Class 4, uncomment the line below to auto-fix it:
                    # if cls == 4: cls = 2

                    x, y, w, h = map(float, parts[1:5])

                    # Clamping prevents out-of-bounds coordinate errors [> 1.0]
                    x = max(0.001, min(0.999, x))
                    y = max(0.001, min(0.999, y))
                    w = max(0.001, min(0.999, w))
                    h = max(0.001, min(0.999, h))

                    boxes.append([x, y, w, h])
                    classes.append(cls)
    return boxes, classes

def save_yolo_labels(boxes, classes, output_path):
    with open(output_path, 'w') as f:
        for box, cls in zip(boxes, classes):
            x, y, w, h = box
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def get_label_path(img_path_str):
    img_path = Path(img_path_str)
    base_name = img_path.stem
    if 'images' in img_path.parts:
        lbl_dir = str(img_path.parent).replace('images', 'labels')
        cand = os.path.join(lbl_dir, base_name + ".txt")
        if os.path.exists(cand): return cand
    cand_flat = os.path.join(img_path.parent, base_name + ".txt")
    return cand_flat

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def generate_yaml():
    yaml_data = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    with open(os.path.join(OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

def compile_dataset():
    setup_directories()
    generate_yaml()
    colorblind_aug = get_colorblind_pipeline()

    print("🔍 Scanning all input directories...")

    # 1. Harvest Patch Data First (Highest Priority)
    patch_pos_imgs = [f for f in glob.glob(os.path.join(PATCH_POS_DIR, "*.*")) if
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(PATCH_POS_DIR) else []
    patch_neg_imgs = [f for f in glob.glob(os.path.join(PATCH_NEG_DIR, "*.*")) if
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(PATCH_NEG_DIR) else []

    # 2. Harvest Synthetic Bulk
    syn_train_imgs = glob.glob(os.path.join(SYNTHETIC_DIR, "images", "train", "*.*")) if os.path.exists(
        SYNTHETIC_DIR) else []
    syn_val_imgs = glob.glob(os.path.join(SYNTHETIC_DIR, "images", "val", "*.*")) if os.path.exists(
        SYNTHETIC_DIR) else []
    syn_imgs = [f for f in (syn_train_imgs + syn_val_imgs) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(
        f"✔️ Found {len(patch_pos_imgs)} Real Positives | {len(patch_neg_imgs)} Real Negatives | {len(syn_imgs)} Synthetic Bulk")

    # Categorize into class buckets
    class_buckets = defaultdict(list)
    negative_bucket = []

    # Process Patch Positives
    for img_p in patch_pos_imgs:
        lbl_p = get_label_path(img_p)
        boxes, classes = read_and_clamp_labels(lbl_p)
        if boxes:
            for c in set(classes):
                if 0 <= c <= 3:
                    class_buckets[c].append((img_p, lbl_p, True))  # True = High Priority Patch

    # Process Patch Negatives
    for img_p in patch_neg_imgs:
        lbl_p = get_label_path(img_p)
        negative_bucket.append((img_p, lbl_p, True))

    # Process Synthetic Bulk
    random.shuffle(syn_imgs)
    for img_p in syn_imgs:
        lbl_p = get_label_path(img_p)
        boxes, classes = read_and_clamp_labels(lbl_p)
        if not boxes:
            negative_bucket.append((img_p, lbl_p, False))
        else:
            for c in set(classes):
                if 0 <= c <= 3:
                    class_buckets[c].append((img_p, lbl_p, False))

    print("\n📊 Available Pool Per Class:")
    for c_id in range(4):
        patch_count = sum(1 for item in class_buckets[c_id] if item[2])
        total_count = len(class_buckets[c_id])
        print(f"   Class {c_id} ({CLASSES[c_id]}): {total_count} total ({patch_count} real patches)")
    print(f"   Hard Negatives: {len(negative_bucket)} total ({len(patch_neg_imgs)} real patches)")

    final_pool = []

    # 3. Assemble Positives Quota (800 per class)
    for c_id in range(4):
        items = class_buckets[c_id]
        if not items:
            print(f"⚠️ WARNING: Class {c_id} ({CLASSES[c_id]}) has 0 items! Check patch labels or asset folder names.")
            continue

        # Sort so real patches (priority = True) are picked first
        items.sort(key=lambda x: x[2], reverse=True)

        if len(items) >= TARGET_PER_CLASS:
            selected = items[:TARGET_PER_CLASS]
            for img_p, lbl_p, _ in selected:
                final_pool.append((img_p, lbl_p, False))
        else:
            # Oversample if under quota
            print(f"  ⚡ Oversampling Class {c_id} ({CLASSES[c_id]}) from {len(items)} to {TARGET_PER_CLASS}...")
            for img_p, lbl_p, _ in items:
                final_pool.append((img_p, lbl_p, False))
            needed = TARGET_PER_CLASS - len(items)
            for _ in range(needed):
                img_p, lbl_p, _ = random.choice(items)
                final_pool.append((img_p, lbl_p, True))  # Mark for colorblind aug

    # 4. Assemble Negatives Quota (1000 total)
    negative_bucket.sort(key=lambda x: x[2], reverse=True)  # Real patch negatives first
    if len(negative_bucket) >= TARGET_NEGATIVES:
        selected_negs = negative_bucket[:TARGET_NEGATIVES]
        for img_p, lbl_p, _ in selected_negs:
            final_pool.append((img_p, lbl_p, False))
    else:
        print(f"  ⚡ Oversampling Hard Negatives from {len(negative_bucket)} to {TARGET_NEGATIVES}...")
        for img_p, lbl_p, _ in negative_bucket:
            final_pool.append((img_p, lbl_p, False))
        needed = TARGET_NEGATIVES - len(negative_bucket)
        for _ in range(needed):
            img_p, lbl_p, _ = random.choice(negative_bucket)
            final_pool.append((img_p, lbl_p, True))

    random.shuffle(final_pool)

    # 5. Split 80/20 and Write
    split_idx = int(len(final_pool) * SPLIT_RATIO)
    train_pool = final_pool[:split_idx]
    val_pool = final_pool[split_idx:]

    print(f"\n📦 Final Compiled Budget: {len(final_pool)} images | Train: {len(train_pool)} | Val: {len(val_pool)}")

    def write_pool(pool, split):
        for idx, (img_path, lbl_path, needs_aug) in enumerate(pool):
            base_name = f"{split}_{idx:05d}"
            out_img_p = os.path.join(OUTPUT_DIR, 'images', split, f"{base_name}.jpg")
            out_lbl_p = os.path.join(OUTPUT_DIR, 'labels', split, f"{base_name}.txt")

            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, classes = read_and_clamp_labels(lbl_path)

            if split == 'train' and (needs_aug or random.random() < 0.4):
                if not boxes:  # Negative image
                    aug = colorblind_aug(image=img, bboxes=[], class_labels=[])
                    cv2.imwrite(out_img_p, cv2.cvtColor(aug['image'], cv2.COLOR_RGB2BGR))
                    open(out_lbl_p, 'w').close()
                else:
                    try:
                        aug = colorblind_aug(image=img, bboxes=boxes, class_labels=classes)
                        if len(aug['bboxes']) > 0:
                            cv2.imwrite(out_img_p, cv2.cvtColor(aug['image'], cv2.COLOR_RGB2BGR))
                            save_yolo_labels(aug['bboxes'], aug['class_labels'], out_lbl_p)
                        else:
                            cv2.imwrite(out_img_p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                            save_yolo_labels(boxes, classes, out_lbl_p)
                    except Exception:
                        cv2.imwrite(out_img_p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        save_yolo_labels(boxes, classes, out_lbl_p)
            else:
                cv2.imwrite(out_img_p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if not boxes:
                    open(out_lbl_p, 'w').close()
                else:
                    save_yolo_labels(boxes, classes, out_lbl_p)

    print("💾 Writing Train split...")
    write_pool(train_pool, 'train')
    print("💾 Writing Val split...")
    write_pool(val_pool, 'val')

    print(f"\n✅ Stitching Complete! Dataset generated at: {OUTPUT_DIR}")

if __name__ == "__main__":
    compile_dataset()