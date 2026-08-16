from ultralytics import YOLO

def main():
    # Start fresh from COCO weights
    model = YOLO('yolov8s.pt')

    print("🚀 Initiating Clean Build from COCO...")

    results = model.train(
        data='deepocean_nuclear.yaml',
        epochs=150,                     # Give it time to learn the domain
        imgsz=640,
        batch=16,
        device=0,
        lr0=0.01,                       # Standard starting learning rate
        optimizer='auto',
        project='DeepOcean_R26_Clean',
        name='R26_Native_4Class',
        patience=25
    )

if __name__ == '__main__':
    main()