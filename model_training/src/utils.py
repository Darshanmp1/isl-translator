import torch
import matplotlib.pyplot as plt
import os

def save_model(model, path):
    """Save PyTorch model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved at {path}")

def plot_loss_curve(losses, save_path="results/loss_curve.png"):
    """Plot training loss curve."""
    plt.figure(figsize=(8,5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Loss curve saved at {save_path}")

def plot_accuracy_curve(accs, save_path="results/accuracy_curve.png"):
    """Plot accuracy curve."""
    plt.figure(figsize=(8,5))
    plt.plot(accs, label="Training Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Accuracy curve saved at {save_path}")
