import os
import cv2
import scipy.io as sio
from pathlib import Path

def convert_shanghaitech_to_yolo(img_dir, gt_dir, output_dir):
    """
    Converts ShanghaiTech .mat ground truth to YOLO .txt labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_extensions = ('.jpg', '.jpeg', '.png')
    
    if not os.path.exists(img_dir):
        print(f"ERROR: Image source directory not found: {img_dir}")
    # Diagnostic: Check if directories exist
    if not os.path.isdir(img_dir):
        print(f"❌ ERROR: Image directory not found at: {img_dir}")
        return
    if not os.path.isdir(gt_dir):
        print(f"❌ ERROR: Ground truth (.mat) directory not found at: {gt_dir}")
        return

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    print(f"Found {len(images)} images. Converting labels...")
    if not images:
        print(f"❌ ERROR: No images found in {img_dir}")
        return

    print(f"Found {len(images)} images. Starting conversion to YOLO format...")

    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # ShanghaiTech naming: IMG_1.jpg -> GT_IMG_1.mat
        mat_name = "GT_" + os.path.splitext(img_name)[0] + ".mat"
        mat_path = os.path.join(gt_dir, mat_name)

        if not os.path.exists(mat_path):
            print(f"  Skipping {img_name}: Ground truth file {mat_name} not found.")
            continue

        # Load .mat file
        # Using uint16 for coordinate data often found in these mat files
        mat_data = sio.loadmat(mat_path)
        # Coordinates are usually in image_info[0,0][0,0][0]
        points = mat_data['image_info'][0,0][0,0][0]

        yolo_labels = []
        for pt in points:
            px, py = pt[0], pt[1]
            
            # Normalize coordinates (0-1)
            x_center = px / w
            y_center = py / h
            # Use a small fixed box size for head detection (e.g., 2% of image)
            bw = 0.02
            bh = 0.02
            
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        # Save .txt file
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(output_dir, txt_name), 'w') as f:
            f.write("\n".join(yolo_labels))
        
        # Also copy the image to the output dir so everything is in one place for split_dataset.py
        cv2.imwrite(os.path.join(output_dir, img_name), img)

    processed_count = len(os.listdir(output_dir)) // 2  # Each image has a .txt
    print(f"Finished! Successfully converted {processed_count} images.")
    print(f"Processed data is in: {output_dir}")

if __name__ == "__main__":
    # IMPORTANT: Verify these paths match your actual folder structure
    root = r"C:\Users\HP\Downloads\archive (3)\ShanghaiTech\part_A\train_data"
    
    img_src = os.path.join(root, "images")
    # Note: Some versions of the dataset use 'ground-truth' or 'ground_truth'
    gt_src = os.path.join(root, "ground_truth") 
    
    # Attempt to find the ground truth folder (tries both underscore and hyphen)
    gt_src = os.path.join(root, "ground_truth")
    if not os.path.exists(gt_src):
        gt_src = os.path.join(root, "ground-truth")
    
    # This is where we will store the YOLO-ready files before splitting
    dest = r"C:\Users\HP\CrowdMonitor\raw_shanghaitech"
    
    convert_shanghaitech_to_yolo(img_src, gt_src, dest)