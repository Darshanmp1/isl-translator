# src/evaluate.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from dataset_loader import ISLDataset
from model import TextToVideoModel

# ============================
# Device
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Paths
# ============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS_CSV = os.path.join(BASE_DIR, "data", "labels.csv")
FRAMES_DIR = os.path.join(BASE_DIR, "data", "processed_frames")
MODEL_PATH = os.path.join(BASE_DIR, "models", "text_to_video_model.pt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# Evaluation Function
# ============================
def evaluate_model():
    # Tokenizer & Dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = ISLDataset(LABELS_CSV, FRAMES_DIR, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Load model
    model = TextToVideoModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.CosineEmbeddingLoss()
    total_loss = 0
    similarities = []
    correct = 0
    total = 0

    with torch.no_grad():
        for text_inputs, video_inputs in dataloader:
            video_inputs = video_inputs.to(device)

            text_inputs = {k: v.squeeze(1).to(device) for k, v in text_inputs.items() if k != "token_type_ids"}

            # Forward pass
            text_feat, video_feat = model(text_inputs, video_inputs)

            # Loss
            labels = torch.ones(text_feat.size(0)).to(device)
            loss = criterion(text_feat, video_feat, labels)
            total_loss += loss.item()

            # Cosine similarity
            cos_sim = nn.functional.cosine_similarity(text_feat, video_feat)
            similarities.extend(cos_sim.cpu().numpy())

            # ----- ACCURACY -----
            preds = (cos_sim > 0.5).long()         # predicted matches
            true_labels = torch.ones_like(preds)    # real matches
            correct += (preds == true_labels).sum().item()
            total += preds.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_similarity = sum(similarities) / len(similarities)
    accuracy = correct / total

    print(f"✅ Evaluation Results:")
    print(f"Average Cosine Loss: {avg_loss:.4f}")
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Plot histogram
    plt.figure(figsize=(8,6))
    plt.hist(similarities, bins=20, color='skyblue', edgecolor='black')
    plt.title("Cosine Similarities Between Text & Video Embeddings")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    hist_path = os.path.join(RESULTS_DIR, "similarity_histogram.png")
    plt.savefig(hist_path)
    print(f"Histogram saved at {hist_path}")


# ============================
# Main
# ============================
if __name__ == "__main__":
    evaluate_model()
