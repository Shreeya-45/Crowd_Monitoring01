import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO, settings
from config import DEVICE

def train_custom_model():
    # 1. Load the base model
    # Suggesting Nano model (yolov8n) for CPU training as it is much faster
    model_type = "yolov8n.pt" if DEVICE == "cpu" else "yolov8m.pt"
    model = YOLO(model_type)

    # 2. Start training
    # 'data' points to your YAML file
    # 'epochs' is how many times the model sees the whole dataset
    # 'imgsz' is the image size (usually 640)
    
    # Resolve absolute path relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Suggest using a relative path or environment variable for portability
    data_path = os.path.join(base_dir, "datasets", "crowd_data", "data.yaml")
    
    dataset_root = os.path.join(base_dir, "datasets", "crowd_data")
    val_labels_dir = os.path.join(dataset_root, "labels", "val")
    val_images_dir = os.path.join(dataset_root, "images", "val")
    train_images_dir = os.path.join(dataset_root, "images", "train")
    train_labels_dir = os.path.join(dataset_root, "labels", "train")

    # Force Ultralytics to use your project's dataset directory
    settings.update({'datasets_dir': os.path.join(base_dir, "datasets")})

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset configuration file not found at: {data_path}\n"
                                "Please ensure you have created the folder and the 'data.yaml' file.")

    # Verify that validation images exist
    if not os.path.exists(val_images_dir) or not any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(val_images_dir)):
        print(f"\nERROR: No images found in {val_images_dir}")
        print("Please run setup_dataset.py and move your labeled images into the train/val folders.")
        return

    # Check if training data exists
    if not os.path.exists(train_images_dir) or not os.listdir(train_images_dir):
        print(f"\nERROR: Training images directory is empty: {train_images_dir}")
        return

    if not os.path.exists(train_labels_dir) or not any(f.endswith('.txt') for f in os.listdir(train_labels_dir)):
        print(f"\nERROR: No .txt label files found in {train_labels_dir}")
        print("Training cannot start without labels.")
        return

    print(f"Using dataset config: {data_path}")
    print(f"Training on device: {DEVICE}")

    # 2. Start training
    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=16,
        name="crowd_monitor_finetuned",
        device=DEVICE,
        exist_ok=True # Overwrite existing run with same name
    )

    # 3. Test/Validate the model
    print("\n--- Training Complete. Starting Validation (Testing) ---")
    # Ensure we validate using the best weights found during training
    metrics = model.val(data=data_path, split='val')
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")
    print(f"Best weights saved to: runs/detect/crowd_monitor_finetuned/weights/best.pt")
    print(f"Model testing complete. Best weights available at: {os.path.join(base_dir, 'runs/detect/crowd_monitor_finetuned/weights/best.pt')}")

if __name__ == "__main__":
    train_custom_model()