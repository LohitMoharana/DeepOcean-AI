import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import albumentations as A

# --- 1. CONFIGURATION ---
NUM_IMAGES = 10000
OUTPUT_DIR = "../dataset_R26_clean"
BG_DIR = "../raw_data/negatives/backgrounds"
ASSET_DIR = "../raw_data/assets"
FINAL_IMG_SIZE = (640, 640)

# R26 Taxonomy (4-Class System)
CLASS_MAP = {
    "sea_mine": 0,
    "underwater_vehicle": 1,
    "diver": 2,
    "torpedo": 3,
    "explosive": 3
}

# Ratios
RATIO_STICKER = 0.35
RATIO_PHYSICS = 0.35
RATIO_EMPTY = 0.30
PROB_ALIEN = 0.95


# --- 2. GEOMETRY ENGINE ---
def safe_rotate(image, angle, border_mode=cv2.BORDER_REFLECT):
    h, w = image.shape[:2]
    angle_rad = np.radians(angle)
    cos_theta = np.abs(np.cos(angle_rad))
    sin_theta = np.abs(np.sin(angle_rad))
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += (new_w / 2) - (w / 2)
    M[1, 2] += (new_h / 2) - (h / 2)
    return cv2.warpAffine(image, M, (new_w, new_h), borderMode=border_mode)


def apply_random_mirror(image):
    if random.random() < 0.5: image = cv2.flip(image, 1)
    if random.random() < 0.5: image = cv2.flip(image, 0)
    return image


# --- 3. AUGMENTATION ENGINE ---
def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def apply_elastic(img):
    transform = A.Compose([
        A.ElasticTransform(p=1.0, alpha=100, sigma=10),
        A.Affine(shear=(-10, 10), p=0.5),
        A.Perspective(p=0.5, scale=(0.05, 0.1))
    ])
    return transform(image=img)['image']


def generate_caustics(shape):
    h, w = shape[:2]
    noise = np.zeros((h, w), dtype=np.uint8)
    cv2.randn(noise, 128, 50)
    noise = cv2.GaussianBlur(noise, (21, 21), 0)
    _, thresh = cv2.threshold(noise, 160, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    caustics = cv2.dilate(thresh, kernel, iterations=1)
    caustics = cv2.GaussianBlur(caustics, (5, 5), 0)
    caustics = cv2.resize(caustics, (w, h))
    return cv2.cvtColor(caustics, cv2.COLOR_GRAY2BGR)


def add_occlusion(img):
    h, w = img.shape[:2]
    if random.random() > 0.5: return img
    num_blobs = random.randint(1, 3)
    for _ in range(num_blobs):
        bx, by = random.randint(0, w), random.randint(0, h)
        br = random.randint(int(min(h, w) * 0.1), int(min(h, w) * 0.4))
        color = np.random.randint(0, 50, (3,)).tolist()
        cv2.circle(img, (bx, by), br, color, -1)
    return img


# --- 4. BLENDING ENGINE (PHYSICS AWARE) ---
def overlay_physics(background, foreground, alpha_mask, x, y, apply_physics=False):
    h_fg, w_fg = foreground.shape[:2]
    h_bg, w_bg = background.shape[:2]
    if x >= w_bg or y >= h_bg: return background
    h_part = min(h_fg, h_bg - y)
    w_part = min(w_fg, w_bg - x)
    if h_part <= 0 or w_part <= 0: return background

    fg_crop = foreground[:h_part, :w_part]
    bg_crop = background[y:y + h_part, x:x + w_part]
    alpha_crop = alpha_mask[:h_part, :w_part]

    if apply_physics:
        # Optical Physics Equation
        z = random.uniform(0.5, 1.5)
        beta = 0.4
        t = np.exp(-beta * z)

        # Calculate local ambient light from the background patch
        A_c = np.mean(bg_crop, axis=(0, 1))

        # Attenuate RGB channels natively
        fg_crop = (fg_crop.astype(float) * t) + (A_c * (1 - t))
        fg_crop = np.clip(fg_crop, 0, 255)

    # Standard Alpha Blending (Strictly locking asset transparency)
    alpha_factor = (alpha_crop.astype(float) / 255.0)
    alpha_factor = np.dstack([alpha_factor] * 3)

    composite = (fg_crop.astype(float) * alpha_factor) + (bg_crop.astype(float) * (1.0 - alpha_factor))
    background[y:y + h_part, x:x + w_part] = composite.astype(np.uint8)
    return background


# --- 5. MAIN PIPELINE ---
def main():
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    bg_files = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    asset_files = []
    for root, dirs, files in os.walk(ASSET_DIR):
        for file in files:
            if file.lower().endswith('.png'):
                asset_files.append(os.path.join(root, file))

    if not bg_files:
        print(f"CRITICAL ERROR: No backgrounds found in {BG_DIR}")
        return
    if not asset_files:
        print(f"CRITICAL ERROR: No assets found in subfolders of {ASSET_DIR}")
        return

    print(f"Generating {NUM_IMAGES} Physics-Aware Frames...")

    for i in tqdm(range(NUM_IMAGES)):
        split = "train" if random.random() < 0.8 else "val"
        r_mode = random.random()
        if r_mode < RATIO_EMPTY:
            mode = "EMPTY"
        elif r_mode < RATIO_EMPTY + RATIO_STICKER:
            mode = "STICKER"
        else:
            mode = "PHYSICS"
        is_alien = random.random() < PROB_ALIEN

        # --- B. BACKGROUND ---
        bg_path = random.choice(bg_files)
        bg = cv2.imread(bg_path)
        if bg is None: continue

        bg = apply_random_mirror(bg)
        bg = safe_rotate(bg, random.uniform(0, 360), border_mode=cv2.BORDER_REFLECT)
        if is_alien:
            bg = augment_hsv(bg, h_gain=1.0, s_gain=0.5, v_gain=0.2)

        # Apply Turbidity (Blur) to the environment BEFORE pasting assets
        if random.random() < 0.5:
            k = random.choice([3, 5, 7])
            bg = cv2.GaussianBlur(bg, (k, k), 0)

        bg_h, bg_w = bg.shape[:2]
        labels = []

        # --- C. ASSET ---
        if mode != "EMPTY":
            asset_path = random.choice(asset_files)
            parent_folder = os.path.basename(os.path.dirname(asset_path))
            class_id = CLASS_MAP.get(parent_folder, -1)

            if class_id != -1:
                asset = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
                if asset is not None:
                    if asset.shape[2] == 4:
                        b, g, r, a = cv2.split(asset)
                        asset_rgb = cv2.merge((b, g, r))
                        alpha = a
                    else:
                        asset_rgb = asset
                        alpha = np.ones(asset.shape[:2], dtype=np.uint8) * 255

                    if random.random() < 0.5:
                        asset_rgb = cv2.flip(asset_rgb, 1)
                        alpha = cv2.flip(alpha, 1)
                    if random.random() < 0.5:
                        asset_rgb = cv2.flip(asset_rgb, 0)
                        alpha = cv2.flip(alpha, 0)

                    if is_alien:
                        asset_rgb = apply_elastic(asset_rgb)

                    rot_angle = random.uniform(0, 360)
                    asset_rgb = safe_rotate(asset_rgb, rot_angle, border_mode=cv2.BORDER_CONSTANT)
                    alpha = safe_rotate(alpha, rot_angle, border_mode=cv2.BORDER_CONSTANT)

                    apply_physics = False
                    if mode == "PHYSICS":
                        apply_physics = True
                    elif mode == "STICKER" and is_alien:
                        asset_rgb = augment_hsv(asset_rgb, h_gain=0.8, s_gain=0.8, v_gain=0.5)

                    asset_rgb = add_occlusion(asset_rgb)

                    ah, aw = asset_rgb.shape[:2]
                    scale = random.uniform(0.15, 0.45)
                    new_w = int(bg_w * scale)
                    new_h = int(ah * (new_w / aw))

                    if new_w > 10 and new_h > 10:
                        asset_rgb = cv2.resize(asset_rgb, (new_w, new_h))
                        alpha = cv2.resize(alpha, (new_w, new_h))

                        max_x = max(0, bg_w - new_w)
                        max_y = max(0, bg_h - new_h)
                        x_pos = random.randint(0, max_x)
                        y_pos = random.randint(0, max_y)

                        bg = overlay_physics(bg, asset_rgb, alpha, x_pos, y_pos, apply_physics=apply_physics)

                        x_c = (x_pos + new_w / 2) / bg_w
                        y_c = (y_pos + new_h / 2) / bg_h
                        w_n = new_w / bg_w
                        h_n = new_h / bg_h
                        labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        # --- D. GLOBAL NOISE ---
        c_intensity = 0.6 if is_alien else 0.3
        caustics = generate_caustics((bg_h, bg_w))
        bg = cv2.addWeighted(bg, 1.0, caustics, c_intensity, 0)

        if random.random() < 0.3:
            noise = np.random.normal(0, 15, bg.shape).astype(np.uint8)
            bg = cv2.add(bg, noise)

        final_img = cv2.resize(bg, FINAL_IMG_SIZE)
        filename = f"{i:05d}_{mode}_{'ALIEN' if is_alien else 'REAL'}"
        img_path = f"{OUTPUT_DIR}/images/{split}/{filename}.jpg"
        lbl_path = f"{OUTPUT_DIR}/labels/{split}/{filename}.txt"

        cv2.imwrite(img_path, final_img)
        with open(lbl_path, "w") as f:
            if labels:
                f.write("\n".join(labels))


if __name__ == "__main__":
    main()