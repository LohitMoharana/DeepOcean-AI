import cv2
from ultralytics import YOLO
import os
import numpy as np
from collections import defaultdict, deque

# --- CONFIGURATION ---
MODEL_PATH = r'../weights/deepocean_final.pt'
OUTPUT_DIR = "../docs/outputs/"
TRACKER_CFG = "custom_botsort.yaml"

# 1. MANUAL VIDEO LIST
VIDEOS = [
    r"../docs/test_videos/test_video_1.",
]

# 2. TWO-STAGE CONFIDENCE THRESHOLDS

# STAGE 1: High confidence required to INITIALIZE tracking (Eliminates false positive flashes)
CONF_INIT_THRESH = {
    0: 0.65,  # Sea Mine must hit 65%+ to start tracking
    1: 0.70,  # UUV must hit 70%+ to start
    2: 0.70,  # Diver must hit 60%+ to start
    3: 0.85   # Misc
}

# STAGE 2: Lower confidence allowed to HOLD/MAINTAIN an established track (Handles zoom/camouflage)
CONF_HOLD_THRESH = {
    0: 0.20,  # Sea Mine hold floor (Dropped from 0.35 to stick through camo)
    1: 0.45,  # UUV hold floor
    2: 0.30,  # Diver hold floor (Keep this low so we don't lose real divers)
    3: 0.60   # Misc hold floor
}

# 3. PERSISTENCE THRESHOLDS (Anti-Flicker Streak)
PERSISTENCE_THRESHOLD = {
    0: 3,  # Sea Mine: Wait 3 frames
    1: 3,  # UUV: Wait 3 frames
    2: 2,  # Diver: Wait 2 frames (Safety Priority)
}

# 4. CLASS NAMES & COLORS
CLASS_NAMES = {
    0: 'SEA MINE',
    1: 'UUV',
    2: 'DIVER (HUMAN)',
    3: 'MISC'
}

COLORS = {
    0: (50, 50, 255),  # Crimson Red
    1: (50, 205, 50),  # Lime Green
    2: (255, 0, 255), # Neon Purple
    3: (128, 128, 128) # Gray
}

# 5. GLOBAL MEMORY BUFFERS
id_history = defaultdict(lambda: deque(maxlen=30))
species_lock = {}
id_persistence = defaultdict(int)
initialized_ids = set()  # Track IDs that passed Stage 1 initialization


# --- LOGIC: SPECIES LOCK ---
def resolve_species(track_id, current_class):
    id_history[track_id].append(current_class)

    counts = defaultdict(int)
    for c in id_history[track_id]:
        counts[c] += 1
    majority_cls = max(counts, key=counts.get)

    if track_id not in species_lock:
        if len(id_history[track_id]) >= 3:
            if majority_cls in [0, 1, 3]:  # Machines/Objects
                species_lock[track_id] = "MACHINE"
            elif majority_cls == 2:  # Human Diver
                species_lock[track_id] = "HUMAN"

    final_cls = majority_cls

    if track_id in species_lock:
        lock_type = species_lock[track_id]
        if lock_type == "MACHINE" and final_cls == 2:
            machine_candidates = [c for c in id_history[track_id] if c != 2]
            if machine_candidates:
                final_cls = max(set(machine_candidates), key=machine_candidates.count)
            else:
                final_cls = 1  # Default to UUV

    return final_cls


# --- DRAWING FUNCTION ---
def draw_professional_label(img, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1
    box_thickness = 2
    padding = 5

    cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
    (text_w, text_h), baseline = cv2.getTextSize(label, font_face, font_scale, font_thickness)

    label_bg_x1 = x1
    label_bg_y1 = y1 - text_h - (padding * 2) - baseline
    label_bg_x2 = x1 + text_w + (padding * 2)
    label_bg_y2 = y1

    text_x = x1 + padding
    text_y = y1 - padding - baseline

    img_h, img_w, _ = img.shape
    if label_bg_y1 < 0:
        label_bg_y1 = y1
        label_bg_y2 = y1 + text_h + (padding * 2) + baseline
        text_y = y1 + text_h + padding + baseline

    label_bg_x2 = min(label_bg_x2, img_w)

    cv2.rectangle(img, (int(label_bg_x1), int(label_bg_y1)), (int(label_bg_x2), int(label_bg_y2)), color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (int(text_x), int(text_y)), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


# --- MAIN PROCESSING LOOP ---
def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    # RESET ALL MEMORY STATE PER VIDEO (Prevents track pollution across videos)
    id_history.clear()
    species_lock.clear()
    id_persistence.clear()
    initialized_ids.clear()

    video_name = os.path.basename(video_path)
    output_name = os.path.splitext(video_name)[0] + "_R26.avi"
    print(f"--> Processing on GPU: {video_name}")

    try:
        model = YOLO(MODEL_PATH)
        model.to('cuda')  # Force weights to GPU VRAM

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, output_name)

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # GPU Accelerated Tracking with low conf floor to catch candidate boxes
            results = model.track(frame, persist=True, tracker=TRACKER_CFG, conf=0.10, iou=0.40, verbose=False, device=0)
            result = results[0]
            annotated_frame = frame.copy()

            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                clss = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for box, track_id, cls, conf in zip(boxes, ids, clss, confs):

                    # 1. Species Lock Resolution
                    final_cls = resolve_species(track_id, cls)

                    # 2. SILENCE MISC CLASS (Class 3)
                    if final_cls == 3:
                        continue

                    # 3. Persistence Check
                    id_persistence[track_id] += 1
                    streak = id_persistence[track_id]
                    required_streak = PERSISTENCE_THRESHOLD.get(final_cls, 3)

                    # 4. TWO-STAGE THRESHOLD LOGIC
                    init_thresh = CONF_INIT_THRESH.get(final_cls, 0.65)
                    hold_thresh = CONF_HOLD_THRESH.get(final_cls, 0.35)

                    # Stage 1: Validate ID if it hits high confidence
                    if track_id not in initialized_ids:
                        if conf >= init_thresh:
                            initialized_ids.add(track_id)

                    # Stage 2: Render box if initialized AND above hold threshold AND streak met
                    if track_id in initialized_ids:
                        if conf >= hold_thresh and streak >= required_streak:
                            base_name = CLASS_NAMES.get(final_cls, str(final_cls))
                            color = COLORS.get(final_cls, (200, 200, 200))

                            if final_cls == 2:  # Diver Class
                                label = f"⚠️ SAFETY LOCK: {base_name} {int(conf * 100)}%"
                            else:
                                label = f"ID:{track_id} {base_name} {int(conf * 100)}%"

                            draw_professional_label(annotated_frame, box, label, color)

            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"    Processed {frame_count} frames...", end='\r')

        cap.release()
        out.release()
        print(f"\n    Saved: {out_path}\n")

    except Exception as e:
        print(f"    CRASH on {video_name}: {e}")


if __name__ == "__main__":
    if not VIDEOS:
        print("Please provide paths in the VIDEOS list.")
    else:
        for v in VIDEOS:
            process_video(v)