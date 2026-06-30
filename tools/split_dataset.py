import os
import random
import shutil
from pathlib import Path

def split_dataset(source_dir, train_ratio=0.8):
    """
    Splits images and labels from a source directory into the YOLO structure.
    Assumes images and .txt labels have the same filename in the source_dir.
    """
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / "datasets" / "crowd_data"
    img_train_dir = base_path / "images" / "train"
    img_val_dir = base_path / "images" / "val"
    lbl_train_dir = base_path / "labels" / "train"
    lbl_val_dir = base_path / "labels" / "val"

    # Create directories if they don't exist
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all image files
    extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(source_dir) if f.lower().endswith(extensions)]
    
    if not images:
        print(f"No images found in {source_dir}")
        return

    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def move_files(file_list, target_img_dir, target_lbl_dir):
        moved_count = 0
        for img_name in file_list:
            # Copy image
            shutil.copy(os.path.join(source_dir, img_name), target_img_dir / img_name)
            
            # Copy corresponding label if it exists
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path = os.path.join(source_dir, lbl_name)
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, target_lbl_dir / lbl_name)
            moved_count += 1
        return moved_count

    print(f"Moving {len(train_images)} files to train...")
    move_files(train_images, img_train_dir, lbl_train_dir)
    
    print(f"Moving {len(val_images)} files to val...")
    move_files(val_images, img_val_dir, lbl_val_dir)

    print("\nDataset split complete!")
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    print("\nNote: YOLO training requires .txt files in the labels folder to work.")
    print("If these folders are empty, your training will error out later.")

if __name__ == "__main__":
    print("--- Dataset Splitter ---")
    # Example: If your frames are in C:\Users\HP\CrowdMonitor\extracted_frames
    default_source = Path(r"C:\Users\HP\Downloads\crowdhuman-dataset.v1i.yolov8")
    
    prompt = f"Enter source directory [default: {default_source}]: "
    source_input = input(prompt).strip().strip('"').strip("'")
    
    source = source_input if source_input else str(default_source)

    if os.path.exists(source):
        split_dataset(source)
    else:
        print(f"Directory not found: {source}")