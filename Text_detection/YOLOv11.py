import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("Text_detection/yolo11m.pt")

    # Train the model
    model.train(
        data="datasets/yolo_data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="0",  # Use GPU 0
        save_period=10,  # Save every 10 epochs
        val=True,  # Validate during training
        cache=True,  # Cache images for faster training
        patience=20,  # Early stopping patience
        plots=True,  # Plot training results
    )

    model_path = "runs/train/yolov11m_custom/weights/best.pt"
    model = YOLO(model_path)

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")