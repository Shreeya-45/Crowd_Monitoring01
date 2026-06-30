import os
import yaml
from ultralytics import YOLO

def main():
    base_dir = r"C:\Users\Dell\Downloads\Crowd_Monitoring2\Crowd_Monitoring1\datasets\CrowdDataset"
    
    # Create an absolute data.yaml
    data = {
        'train': os.path.join(base_dir, 'train', 'images'),
        'val': os.path.join(base_dir, 'valid', 'images'),
        'test': os.path.join(base_dir, 'test', 'images'),
        'nc': 2,
        'names': ['car', 'person']
    }
    
    abs_yaml_path = os.path.join(base_dir, 'abs_data.yaml')
    with open(abs_yaml_path, 'w') as f:
        yaml.dump(data, f)
        
    model_path = os.path.join(base_dir, 'runs', 'detect', 'runs', 'crowd_detection-7', 'weights', 'best.pt')
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("\n--- Evaluating on Validation Set ---")
    metrics_val = model.val(data=abs_yaml_path, split='val', verbose=False)
    
    print("\n--- Evaluating on Test Set ---")
    try:
        metrics_test = model.val(data=abs_yaml_path, split='test', verbose=False)
    except Exception as e:
        print(f"Could not evaluate on test set (it might not exist or be empty): {e}")

if __name__ == "__main__":
    main()
