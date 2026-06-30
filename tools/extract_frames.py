import cv2
import os

def extract(video_path, output_dir, interval=5):
    """Extracts every 'interval' frame from a video or GIF for training."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    
    print(f"Starting extraction from: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % interval == 0:
            fname = os.path.join(output_dir, f"vid_frame_{saved:05d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
            
        count += 1
        
    cap.release()
    print(f"Successfully extracted {saved} frames to {output_dir}")
    print("\nNEXT STEP: You MUST label these images (create .txt files) before training.")

if __name__ == "__main__":
    v = input("Path to your GIF file: ").strip()
    o = input("Directory to save extracted frames (e.g., ./raw_images): ").strip()
    extract(v, o)