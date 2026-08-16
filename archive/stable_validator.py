import cv2
from ultralytics import YOLO
import os
import numpy as np
from collections import defaultdict, deque

# --- CONFIGURATION ---
MODEL_PATH = 'D:\Projects\Personal\DeepOcean_V2.0/training\DeepOcean_v25_finetune_3/runs\Hybrid_Reality_Run_34/weights/best.pt'
OUTPUT_DIR = "C:/Users/lohit/Downloads/R26_results_final"
TRACKER_CFG = "custom_botsort.yaml"

# 1. MANUAL VIDEO LIST
VIDEOS = [
    r"C:/Users/lohit/Downloads/test_videos/test_video_1.mp4",
    r"C:/Users/lohit/Downloads/test_videos/test_video_2.mp4",
    r"C:/Users/lohit/Downloads/test_videos/test_video_3.mp4",
    # Add other videos here...
]

# 2. CLASS-SPECIFIC THRESHOLDS (Confidence)
THRESHOLDS = {
    0: 0.50,  # Sea Mine
    1: 0.60,  # UUV
    2: 0.35,  # Diver (Was 4, now 2. Low threshold for safety)
}

# --- NEW: PERSISTENCE THRESHOLDS (Anti-Flicker) ---
# Minimum frames a track must exist before we show it.
# Higher = More stable, less flickering. Lower = Faster reaction.
PERSISTENCE_THRESHOLD = {
    0: 5,  # Sea Mine: Wait 5 frames
    1: 5,  # ROV: Wait 5 frames (Fixes the Controller glitch)
    2: 2,  # Diver: Wait only 2 frames (Safety Priority: Show FAST)
}

# 3. CLASS NAMES & COLORS
CLASS_NAMES = {
    0: 'SEA MINE',
    1: 'UUV',
    2: 'DIVER (HUMAN)'
}

COLORS = {
    0: (50, 50, 255),  # Crimson Red
    1: (50, 205, 50),  # Lime Green
    2: (255, 0, 255)   # Neon Purple
}

# 4. MEMORY BUFFERS
id_history = defaultdict(lambda: deque(maxlen=30))
species_lock = {}  # Stores permanent identity (MACHINE vs HUMAN)
id_persistence = defaultdict(int)  # <-- NEW: Tracks how long we've seen an ID


# --- LOGIC: SPECIES LOCK ---
def resolve_species(track_id, current_class):
    """
    Prevents an ID from switching between Human and Machine.
    If it started as a Machine, it stays a Machine.
    """
    # 1. Update History
    id_history[track_id].append(current_class)

    # 2. Determine Majority Class (Visual Stability)
    counts = defaultdict(int)
    for c in id_history[track_id]: counts[c] += 1
    majority_cls = max(counts, key=counts.get)

    # 3. SET THE LOCK (If not set yet)
    if track_id not in species_lock:
        # Wait for 5 frames of history before locking
        if len(id_history[track_id]) >= 5:
            # FIXED: 0(Mine), 1(UUV), 3(Misc) are Machines.
            if majority_cls in [0, 1, 3]:
                species_lock[track_id] = "MACHINE"
            # FIXED: Diver is now Class 2
            elif majority_cls == 2:
                species_lock[track_id] = "HUMAN"

    # 4. ENFORCE THE LOCK
    final_cls = majority_cls

    if track_id in species_lock:
        lock_type = species_lock[track_id]

        # Scenario: Locked as MACHINE, but Model says HUMAN
        # FIXED: Check against Class 2 (Diver)
        if lock_type == "MACHINE" and final_cls == 2:
            # Force it back to the most common machine class seen so far
            machine_candidates = [c for c in id_history[track_id] if c != 2]
            if machine_candidates:
                final_cls = max(set(machine_candidates), key=machine_candidates.count)
            else:
                final_cls = 1  # Default to ROV if purely confused

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

    cv2.rectangle(img, (int(label_bg_x1), int(label_bg_y1)), (int(label_bg_x2), int(label_bg_y2)), color, -1,
                  cv2.LINE_AA)
    cv2.putText(img, label, (int(text_x), int(text_y)), font_face, font_scale, (255, 255, 255), font_thickness,
                cv2.LINE_AA)


# --- MAIN LOOP ---
def process_video(video_path):
    if not os.path.exists(video_path): return
    video_name = os.path.basename(video_path)
    output_name = os.path.splitext(video_name)[0] + "_stabilized.avi"
    print(f"--> Processing: {video_name}")

    try:
        model = YOLO(MODEL_PATH)
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

            results = model.track(frame, persist=True, tracker=TRACKER_CFG, conf=0.1, verbose=False)
            result = results[0]
            annotated_frame = frame.copy()

            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                clss = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for box, track_id, cls, conf in zip(boxes, ids, clss, confs):

                    # 1. APPLY SPECIES LOCK
                    final_cls = resolve_species(track_id, cls)

                    # --- FIXED: THE NUCLEAR SILENCER ---
                    # If the model detects the MISC class (Class 3), skip drawing completely.
                    if final_cls == 3:
                        continue

                    # 2. UPDATE PERSISTENCE (Anti-Flicker Logic)
                    id_persistence[track_id] += 1
                    streak = id_persistence[track_id]
                    required_streak = PERSISTENCE_THRESHOLD.get(final_cls, 5)

                    # 3. CHECK THRESHOLDS
                    req_thresh = THRESHOLDS.get(final_cls, 0.5)

                    # Draw ONLY if Confidence is met AND Persistence is met
                    if conf >= req_thresh and streak >= required_streak:

                        base_name = CLASS_NAMES.get(final_cls, str(final_cls))
                        color = COLORS.get(final_cls, (200, 200, 200))

                        if final_cls == 2:
                            label = f"⚠️ SAFETY LOCK: {base_name} {int(conf * 100)}%"
                        else:
                            label = f"ID:{track_id} {base_name} {int(conf * 100)}%"

                        draw_professional_label(annotated_frame, box, label, color)

            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 50 == 0: print(f"    Processed {frame_count} frames...", end='\r')

        cap.release()
        out.release()
        print(f"\n    Saved: {out_path}\n")

    except Exception as e:
        print(f"    CRASH on {video_name}: {e}")


if __name__ == "__main__":
    if not VIDEOS:
        print("Please uncomment a video path in the VIDEOS list.")
    else:
        for v in VIDEOS:
            process_video(v)