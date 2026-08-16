from ultralytics import YOLO

def main():
    model = YOLO(r'../weights/deepocean_baseline.pt')

    results = model.train(
        data=r'../R26_finetune_dataset\r26_finetune.yaml',
        epochs=35,
        patience=10,
        imgsz=640,
        batch=16,
        device=0,
        freeze=10,
        lr0=0.001,
        optimizer='AdamW',
        project='DeepOcean_R26_Finetune',
        name='V25_Anchored_35Epochs'
    )

if __name__ == '__main__':
    main()