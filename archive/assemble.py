import shutil
import os
import random
import glob
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION: DEFINE YOUR SOURCES ---
# Format: "Unique_Prefix": "Path/To/Folder"
# The script assumes each folder has 'images' and 'labels' subfolders,
# OR just contains mixed images and txt files.
SOURCES = {
    # 1. The Core Knowledge (Your V26 Data)
    # "original_v26": "datasets/DeepOcean_V2/train",

    # 2. The Manual Safety Data (Your 179 real divers)
    "real_divers": "dataset/alien_data/divers",

    # 3. The Alien Factories (The generated 1000s)
    "alien_divers": "dataset/alien_data/class0and1",
    # Uncomment these if you generated them:
    "alien_mines": "dataset/normal",
    # "alien_vehicles": "datasets/alien_data/vehicles",

    # 4. The Backgrounds (Negative Mining)
    # "alien_bg": "datasets/alien_data/backgrounds",
    # "real_bg": "datasets/negatives/backgrounds"
}

# Where to build the final dataset
OUTPUT_DIR = "DeepOcean_Nuclear"
TRAIN_RATIO = 0.8  # 85% Train, 15% Val

# Extensions to look for
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp','.avif','.webp'}


def assemble():
    # 1. Setup Directories
    base = Path(OUTPUT_DIR)
    if base.exists():
        print(f"⚠️ Warning: '{OUTPUT_DIR}' already exists.")
        resp = input("Delete and rebuild? (y/n): ")
        if resp.lower() == 'y':
            shutil.rmtree(base)
        else:
            return

    dirs = {
        "train_img": base / "train" / "images",
        "train_lbl": base / "train" / "labels",
        "val_img": base / "val" / "images",
        "val_lbl": base / "val" / "labels"
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Assembling dataset from {len(SOURCES)} sources...")

    total_images = 0

    # 2. Iterate through each source
    for prefix, source_path in SOURCES.items():
        src = Path(source_path)

        # Find all images recursively
        # This handles both flat folders and 'images/labels' structures
        all_files = []
        for ext in IMG_EXT:
            all_files.extend(src.rglob(f"*{ext}"))

        # Shuffle for random split
        random.shuffle(all_files)

        # Calculate split index
        split_idx = int(len(all_files) * TRAIN_RATIO)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(f"   📂 Processing '{prefix}': Found {len(all_files)} images.")

        # Function to copy files
        def copy_batch(files, img_dest, lbl_dest):
            # The Universal Migration Map (Old -> New)
            LABEL_MAP = {
                '0': '0',  # Mine -> Mine
                '1': '1',  # UUV -> UUV
                '4': '2',  # Diver (Old 4) -> Diver (New 2)
                '2': '3',  # Torpedo -> MISC
                '3': '3',  # Explosive -> MISC
                '5': '3'  # Distractors -> MISC
            }

            for img_path in tqdm(files, desc=f"      Copying to {img_dest.parent.name}", leave=False):
                new_name = f"{prefix}_{img_path.name}"
                shutil.copy2(img_path, img_dest / new_name)

                lbl_path = img_path.with_suffix(".txt")
                if not lbl_path.exists() and "images" in img_path.parts:
                    parts = list(img_path.parts)
                    parts[parts.index("images")] = "labels"
                    lbl_path = Path(*parts).with_suffix(".txt")

                dest_lbl_path = lbl_dest / f"{prefix}_{img_path.stem}.txt"

                if lbl_path.exists():
                    # MIGRATION LOGIC: Read, Map, Write
                    with open(lbl_path, 'r') as f_in, open(dest_lbl_path, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if parts:
                                old_cls = parts[0]
                                new_cls = LABEL_MAP.get(old_cls, '3')  # Default to MISC
                                f_out.write(f"{new_cls} " + " ".join(parts[1:]) + "\n")
                else:
                    with open(dest_lbl_path, "w") as f:
                        pass

        copy_batch(train_files, dirs["train_img"], dirs["train_lbl"])
        copy_batch(val_files, dirs["val_img"], dirs["val_lbl"])

        total_images += len(all_files)

    # 3. Create the data.yaml automatically
    yaml_content = f"""
    path: ../{OUTPUT_DIR.split('/')[-1]} # Relative path from yolov8 execution
    train: train/images
    val: val/images

    nc: 4
    names:
      0: sea_mine
      1: uuv
      2: diver
      3: misc
    """
    with open(base / "deepocean_nuclear.yaml", "w") as f:
        f.write(yaml_content)

    print("-" * 50)
    print(f"✅ Assembly Complete!")
    print(f"Total Images: {total_images}")
    print(f"Dataset Location: {OUTPUT_DIR}")
    print(f"YAML File: {base / 'deepocean_nuclear.yaml'}")
    print("-" * 50)


if __name__ == "__main__":
    assemble()