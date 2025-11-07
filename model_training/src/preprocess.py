# src/preprocess.py
import os
import cv2
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_frames")
LABELS_CSV = os.path.join(BASE_DIR, "data", "labels.csv")

def generate_labels_csv(raw_video_dir, output_csv):
    data = []
    for file in os.listdir(raw_video_dir):
        if file.endswith(('.mp4', '.avi')):
            label = os.path.splitext(file)[0].replace('_', ' ')
            data.append({'video_name': file, 'label': label})
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ labels.csv saved to {output_csv}")

def extract_frames(video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for video_file in os.listdir(video_dir):
        if not video_file.endswith(('.mp4', '.avi')):
            continue
        
        folder_name = os.path.splitext(video_file)[0]
        video_out_dir = os.path.join(output_dir, folder_name)
        os.makedirs(video_out_dir, exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128))
            frame_path = os.path.join(video_out_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
        cap.release()
        print(f"Extracted {count} frames from {video_file} → {video_out_dir}")

if __name__ == "__main__":
    generate_labels_csv(RAW_DIR, LABELS_CSV)
    extract_frames(RAW_DIR, PROCESSED_DIR)
