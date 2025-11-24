"""
Modern ISL Model Trainer (Unified Final Version ✅)
---------------------------------------------------
✅ Loads and merges all .npz datasets (letters, digits, words)
✅ Works with A–Z, 0–9, and any custom words
✅ Removes classes with <2 samples (avoids sklearn errors)
✅ Automatically balances all classes
✅ Trains SVM classifier (StandardScaler + RBF)
✅ Saves trained model + label encoder safely
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter

# ===== PATH CONFIG =====
DATA_DIR = Path("data_npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "classifier.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"


# ===== LOAD & MERGE DATA =====
def load_npz_data(data_dir):
    """Load and merge all .npz files in data_npz."""
    X_list, y_list = [], []
    files = list(data_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError("❌ No .npz dataset files found! Run 01_collect_dataset first.")

    print(f"\n📦 Found {len(files)} dataset(s):")
    for f in files:
        print(f"   ➜ {f.name}")
        data = np.load(f)
        X, y = data["X"], data["y"]
        X_list.append(X)
        y_list.append(y)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"\n✅ Loaded total samples: {X.shape[0]} | Features per sample: {X.shape[1]}")
    return X, y


# ===== CLEAN DATA =====
def clean_dataset(X, y):
    """Remove labels with fewer than 2 samples + remove banned labels."""
    counts = Counter(y)
    print("\n🔍 Original class distribution:")
    for lbl, c in sorted(counts.items()):
        print(f"   {lbl}: {c}")

    # --- 🛑 Banned Classes (Remove Explicitly) ---
    banned_labels = {"FUCK YOU"}   # add more if needed

    # Remove banned labels
    mask_not_banned = ~np.isin(y, list(banned_labels))
    X = X[mask_not_banned]
    y = y[mask_not_banned]

    if banned_labels:
        print(f"\n❌ Removed banned labels: {list(banned_labels)}")

    # Recount after removing banned classes
    counts = Counter(y)

    # --- Remove labels with fewer than 2 samples ---
    valid_labels = [lbl for lbl, c in counts.items() if c >= 2]

    if len(valid_labels) < 2:
        print("⚠️ Not enough valid labels with ≥2 samples. Please collect more data.")
        return None, None

    mask = np.isin(y, valid_labels)
    X_clean, y_clean = X[mask], y[mask]

    removed = [lbl for lbl in counts if lbl not in valid_labels]
    if removed:
        print(f"\n⚠️ Removed labels with <2 samples: {removed}")

    new_counts = Counter(y_clean)
    print("\n✅ Cleaned class distribution:")
    for lbl, c in sorted(new_counts.items()):
        print(f"   {lbl}: {c}")

    return X_clean, y_clean


# ===== TRAIN MODEL =====
def train_model(X, y):
    """Train SVM classifier with scaling + class balancing."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n🔤 Classes Detected: {list(le.classes_)}")
    print(f"🔢 Total Unique Classes: {len(le.classes_)}")

    # Split safely
    stratify = y_enc if len(np.unique(y_enc)) > 1 else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.2, stratify=stratify, random_state=42
        )
    except ValueError:
        print("⚠️ Not enough samples for stratified split — using random split.")
        X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    print("\n🚀 Training SVM model (RBF kernel)...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            probability=True,
            class_weight="balanced"
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"\n✅ Validation Accuracy: {acc:.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0))

    return clf, le


# ===== SAVE MODEL =====
def save_model(clf, le):
    """Save trained model and label encoder."""
    if clf is None:
        print("⚠️ Model not saved — training incomplete.")
        return
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"💾 Label encoder saved to: {ENCODER_PATH}")


# ===== MAIN =====
if __name__ == "__main__":
    X, y = load_npz_data(DATA_DIR)
    X, y = clean_dataset(X, y)

    if X is not None and y is not None:
        clf, le = train_model(X, y)
        save_model(clf, le)
