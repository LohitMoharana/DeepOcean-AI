import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
MANUAL_DIR = Path("DeepOcean_Nuclear")  # Source 1: Your manual frames
MAIN_DATA_DIR = Path("dataset_R26_clean")  # Source 2: Your synthetic data
OUTPUT_DIR = Path("R26_finetune_dataset")  # Destination

# How many 'misc' images to pull to prevent forgetting
MISC_TRAIN_SAMPLES = 150
MISC_VAL_SAMPLES = 30


def create_dirs():
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)


def process_manual_frames(split):
    """Copies manual frames and changes Diver class 4 to 2."""
    img_dir = MANUAL_DIR / split / 'images'
    lbl_dir = MANUAL_DIR / split / 'labels'

    if not img_dir.exists(): return

    print(f"Processing manual {split} frames...")

    for img_file in img_dir.glob("*.*"):
        if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue

        # Copy image
        shutil.copy2(img_file, OUTPUT_DIR / 'images' / split / img_file.name)

        # Fix and copy label
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        out_lbl_file = OUTPUT_DIR / 'labels' / split / lbl_file.name

        if lbl_file.exists():
            with open(lbl_file, 'r') as f:
                lines = f.readlines()

            with open(out_lbl_file, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue

                    # THE FIX: Remap class 4 to 2
                    if (parts[0] == '4') or (parts[0] == '4.0'):
                        parts[0] = '2'

                    f.write(" ".join(parts) + "\n")


def sample_misc_class(split, sample_count):
    """Pulls images containing the 'misc' class (3) from the main dataset."""
    img_dir = MAIN_DATA_DIR / 'images' / split
    lbl_dir = MAIN_DATA_DIR / 'labels' / split

    if not lbl_dir.exists(): return

    print(f"Sampling {sample_count} 'misc' frames for {split}...")

    misc_files = []
    for lbl_file in lbl_dir.glob("*.txt"):
        with open(lbl_file, 'r') as f:
            # Check if class 3 exists in the label file
            if any(line.startswith('3 ') for line in f):
                misc_files.append(lbl_file)

    # Randomly select subset
    random.shuffle(misc_files)
    sampled = misc_files[:sample_count]

    for lbl_file in sampled:
        # Copy label (No changes needed, already 4-class native)
        shutil.copy2(lbl_file, OUTPUT_DIR / 'labels' / split / lbl_file.name)

        # Find and copy matching image
        for ext in ['.jpg', '.png', '.jpeg']:
            img_file = img_dir / f"{lbl_file.stem}{ext}"
            if img_file.exists():
                shutil.copy2(img_file, OUTPUT_DIR / 'images' / split / img_file.name)
                break


def generate_yaml():
    yaml_content = f"""path: ../{OUTPUT_DIR.name}
train: images/train
val: val/train # Using train for val since manual dataset is tiny

nc: 4
names:
  0: sea_mine
  1: uuv
  2: diver
  3: misc
"""
    yaml_path = OUTPUT_DIR / 'r26_finetune.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created YAML configuration at {yaml_path}")


def main():
    if OUTPUT_DIR.exists():
        print(f"Cleaning existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    create_dirs()
    process_manual_frames('train')
    process_manual_frames('val')
    sample_misc_class('train', MISC_TRAIN_SAMPLES)
    sample_misc_class('val', MISC_VAL_SAMPLES)
    generate_yaml()
    print("✅ Targeted Dataset Assembly Complete.")


if __name__ == '__main__':
    main()