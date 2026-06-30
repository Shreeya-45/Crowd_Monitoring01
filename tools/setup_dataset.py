import os
from pathlib import Path

def initialize_dataset():
    # Define the root path for the dataset
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / "datasets" / "crowd_data"
    
    # Required YOLOv8 directory structure
    subdirs = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]
    
    for subdir in subdirs:
        path = base_path / subdir
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")

    # Create or overwrite data.yaml with absolute paths for reliability
    yaml_content = f"""path: {base_path}
train: images/train
val: images/val

names:
  0: person
"""
    with open(base_path / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Updated dataset configuration: {base_path / 'data.yaml'}")

if __name__ == "__main__":
    initialize_dataset()