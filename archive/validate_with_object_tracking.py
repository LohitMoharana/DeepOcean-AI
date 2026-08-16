import cv2
from ultralytics import YOLO
import os
import torch

# --- CONFIGURATION ---
MODEL_PATH = 'D:\Projects\Personal\DeepOcean_V2.0/training\DeepOcean_V25_finetune\DeepOcean_V25_Finetune\Hybrid_Reality_Run/weights/best.pt'
OUTPUT_DIR = "C:/Users/lohit/Downloads/v25_results"

# 1. MANUAL VIDEO LIST (Full Paths)
VIDEOS = [
    r"C:/Users/lohit/Downloads/test_videos/test_video_1.mp4",
    r"C:/Users/lohit/Downloads/test_videos/test_video_2.mp4",
    r"C:/Users/lohit/Downloads/test_videos/test_video_3.mp4",
    # Add other videos here...
]

# 2. CLASS-SPECIFIC THRESHOLDS
# We apply these AFTER the tracker to filter out weak IDs
THRESHOLDS = {
    0: 0.5,  # Sea Mine
    1: 0.60,  # ROV (Bumped to 0.60 to kill the Diver/Explosive glitch)
    2: 0.5,  # Torpedo
    3: 0.5  # Explosive
}


def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"Skipping {os.path.basename(video_path)} (Not found)")
        return

    video_name = os.path.basename(video_path)
    output_name = os.path.splitext(video_name)[0] + "_tracked.avi"
    print(f"--> Processing with BoT-SORT: {video_name}")

    try:
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"    ERROR: Could not open {video_name}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30

        out_path = os.path.join(OUTPUT_DIR, f"V25_Finetune_with_Object_Tracking_{output_name}")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # --- TRACKING LOGIC ---
            # 1. Run Tracking instead of Prediction
            # tracker="botsort.yaml" uses camera motion compensation (good for drones)
            # persist=True maintains ID history across frames
            results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.1, verbose=False)
            result = results[0]

            # 2. FILTER BOXES & IDs MANUALLY
            keep_indices = []
            if len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # Check against our custom dictionary
                    req_thresh = THRESHOLDS.get(cls_id, 0.5)

                    if conf >= req_thresh:
                        keep_indices.append(i)

            # 3. Create a clean result with only passing boxes
            if keep_indices:
                # Filter the boxes (This keeps the ID attached to the box)
                result.boxes = result.boxes[keep_indices]
                annotated_frame = result.plot()  # This will now draw the ID # number too
            else:
                annotated_frame = frame

            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 50 == 0: print(f"    Processed {frame_count} frames...", end='\r')

        cap.release()
        out.release()
        print(f"\n    Saved: {out_path}\n")

    except Exception as e:
        print(f"    CRASH on {video_name}: {e}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting BoT-SORT Tracker Validation...")
    print(f"Thresholds: {THRESHOLDS}")

    for v in VIDEOS:
        process_video(v)