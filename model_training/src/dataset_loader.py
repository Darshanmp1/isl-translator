# src/dataset_loader.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class ISLDataset(Dataset):
    def __init__(self, labels_csv, frames_dir, tokenizer, max_length=20, num_frames=16):
        """
        labels_csv: path to labels.csv (video_name, label)
        frames_dir: directory containing processed frames
        tokenizer: HuggingFace tokenizer for text
        max_length: max token length for text
        num_frames: fixed number of frames per video
        """
        self.df = pd.read_csv(labels_csv)
        self.frames_dir = frames_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_frames = num_frames

        # Transform for video frames
        self.frame_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = row['video_name']
        text_label = row['label']

        # Handle spaces in folder names
        frame_folder = os.path.join(self.frames_dir, os.path.splitext(video_name)[0])
        
        # If folder doesn't exist, return zero tensor and warn
        if not os.path.exists(frame_folder):
            print(f"⚠️ Warning: Missing frames for '{video_name}'. Returning zero tensor.")
            video_tensor = torch.zeros(3, self.num_frames, 128, 128)
        else:
            # List all frame files
            frame_files = sorted([os.path.join(frame_folder, f) 
                                  for f in os.listdir(frame_folder) if f.endswith('.jpg')])

            # Sample or pad frames to fixed length
            if len(frame_files) >= self.num_frames:
                indices = torch.linspace(0, len(frame_files) - 1, steps=self.num_frames).long()
            else:
                indices = list(range(len(frame_files))) + [len(frame_files)-1]*(self.num_frames - len(frame_files))
                indices = torch.tensor(indices)

            frames = [self.frame_transform(Image.open(frame_files[i]).convert("RGB")) for i in indices]
            video_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)

        # Tokenize text and remove batch dim
        text_inputs = self.tokenizer(
            text_label,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        return text_inputs, video_tensor
