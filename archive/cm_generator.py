# from ultralytics import YOLO
#
# # 1. Load your compiled R26 model
# # Update this path if your weights are in a different directory
# model = YOLO("D:\Projects\Personal\DeepOcean_V2.0/training\DeepOcean_R26/runs\detect\DeepOcean_R26_Finetune\V25_Anchored_35Epochs\weights/best.pt")
#
# # 2. Run validation
# # Replace 'deepocean_nuclear.yaml' with whichever yaml contains your R26 validation set paths
# metrics = model.val(
#     data="R26_finetune_dataset/r26_finetune.yaml",
#     split="val",       # Ensures it evaluates on the validation split
#     plots=True,        # Forces the generation of metric plots (including the confusion matrix)
#     conf=0.25,         # Base confidence threshold
#     iou=0.5            # NMS IOU threshold
# )
#
# print("\n✅ Validation complete!")
# print("The confusion matrix images have been saved to your 'runs/detect/val' directory.")
#
# # Optional: If you want to print the raw mathematical array to the terminal
# matrix_array = metrics.confusion_matrix.matrix
# print("\nRaw Confusion Matrix Array:")
# print(matrix_array)

from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load your compiled R26 model
    model = YOLO("D:\Projects\Personal\DeepOcean_V2.0/training\DeepOcean_R26/runs\detect\DeepOcean_R26_Finetune\V25_Anchored_35Epochs\weights/best.pt")

    # 2. Run validation
    metrics = model.val(
        data="R26_finetune_dataset/r26_finetune.yaml",
        split="val",       # Ensures it evaluates on the validation split
        plots=True,        # Forces the generation of metric plots (including the confusion matrix)
        conf=0.25,         # Base confidence threshold
        iou=0.5,           # NMS IOU threshold
        workers=2          # Optional: explicitly sets the number of dataloader workers
    )

    print("\n✅ Validation complete!")
    print("The confusion matrix images have been saved to your 'runs/detect/val' directory.")

    # Optional: If you want to print the raw mathematical array to the terminal
    matrix_array = metrics.confusion_matrix.matrix
    print("\nRaw Confusion Matrix Array:")
    print(matrix_array)