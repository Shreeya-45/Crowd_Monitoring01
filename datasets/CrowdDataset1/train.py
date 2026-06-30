from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project="runs",
        name="crowd_detection"
        
    )

if __name__ == "__main__":
    main()