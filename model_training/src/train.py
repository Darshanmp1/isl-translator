# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset_loader import ISLDataset  # your custom dataset
from model import TextToVideoModel

# ============================
# Paths
# ============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS_CSV = os.path.join(BASE_DIR, "data", "labels.csv")
FRAMES_DIR = os.path.join(BASE_DIR, "data", "processed_frames")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================
# Hyperparameters
# ============================
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4
MAX_SEQ_LEN = 20

# ============================
# Training Function
# ============================
def train_model():
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Dataset & DataLoader
    dataset = ISLDataset(LABELS_CSV, FRAMES_DIR, tokenizer, max_length=MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Device & model (CPU only)
    device = torch.device("cpu")
    model = TextToVideoModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CosineEmbeddingLoss()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for text_inputs, video_inputs in dataloader:
            # Skip batches with zero video tensors
            if video_inputs.sum() == 0:
                print("⚠️ Skipping batch due to missing frames")
                continue

            # Move video tensor to device
            video_inputs = video_inputs.to(device)

            # Tokenizer outputs: move each tensor to device and squeeze extra dim
            text_inputs = {k: v.squeeze(1).to(device) for k, v in text_inputs.items()}

            # Remove token_type_ids if present (some models don't use it)
            text_inputs = {k: v for k, v in text_inputs.items() if k != "token_type_ids"}

            # Labels for CosineEmbeddingLoss
            labels = torch.ones(video_inputs.size(0)).to(device)

            # Forward pass
            text_feat, video_feat = model(text_inputs, video_inputs)
            loss = criterion(text_feat, video_feat, labels)
            epoch_loss += loss.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint each epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"text_to_video_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(BASE_DIR, "models", "text_to_video_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved at: {final_model_path}")

# ============================
# Main
# ============================
if __name__ == "__main__":
    train_model()
