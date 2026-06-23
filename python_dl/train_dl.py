import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models

# Import from our optimized processed.py
from processed import extract_image, extract_texture_features

DATASET_PATH = "dataset/PlantVillage"
MODEL_PATH = "model/hybrid_leaf_model.pth"
CLASS_PATH = "model/class_names.json"

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Using Device: {device}")

# --------------------------
# OPTIMIZED DATASET
# --------------------------
class PlantDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(root))

        print("[INFO] Loading dataset paths...")
        for idx, cls in enumerate(self.classes):
            folder = os.path.join(root, cls)
            for file in os.listdir(folder):
                self.samples.append(os.path.join(folder, file))
                self.labels.append(idx)
        
        # CPU Bottleneck Fix: Cache handcrafted features in memory
        # This takes up very little RAM but speeds up training exponentially
        self.texture_cache = {}
        print(f"[INFO] Found {len(self.samples)} images across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]

        # 1. Standard CNN Image Processing
        img = extract_image(path)

        # 2. Handcrafted Texture Feature Fetching (with caching)
        if idx in self.texture_cache:
            tex = self.texture_cache[idx]
        else:
            tex = extract_texture_features(path)
            self.texture_cache[idx] = tex # Cache for future epochs

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(tex, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

# --------------------------
# HYBRID ARCHITECTURE
# --------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes, texture_dim):
        super().__init__()

        # --- CNN Branch (Global Spatial Features) ---
        base = models.mobilenet_v2(weights="IMAGENET1K_V1")
        base.classifier = nn.Identity()
        self.cnn = base

        # Freeze early layers to retain generic feature extractors and prevent overfitting
        for param in self.cnn.features[:14].parameters():
            param.requires_grad = False

        self.cnn_norm = nn.LayerNorm(1280)
        self.cnn_proj = nn.Linear(1280, 256)

        # --- Handcrafted Branch (Local Texture & Color) ---
        self.texture_net = nn.Sequential(
            nn.Linear(texture_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # --- Late Fusion Gate (Attention Mechanism) ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(384, 128), # 256 (CNN) + 128 (Texture) = 384
            nn.ReLU(),
            nn.Linear(128, 384),
            nn.Sigmoid()
        )

        # --- Final Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, tex):
        # Process CNN features
        cnn_feat = self.cnn(img)
        cnn_feat = self.cnn_norm(cnn_feat)
        cnn_feat = self.cnn_proj(cnn_feat)

        # Process Handcrafted features
        tex_feat = self.texture_net(tex)

        # Concatenate
        fused = torch.cat([cnn_feat, tex_feat], dim=1)

        # Apply attention gate
        gate = self.fusion_gate(fused)
        fused = fused * gate

        # Classify
        return self.classifier(fused)

# --------------------------
# TRAINING PIPELINE
# --------------------------
def train():
    dataset = PlantDataset(DATASET_PATH)

    # Ensure model directory exists and save class names
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(CLASS_PATH, "w") as f:
        json.dump(dataset.classes, f)

    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Dynamically grab the dimension of the texture vector from the first sample
    texture_dim = dataset[0][1].shape[0]

    model = HybridModel(len(dataset.classes), texture_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Reduce learning rate when validation accuracy plateaus
    # Reduce learning rate when validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        if epoch == 0:
            print("[INFO] Epoch 1 will be slower as it builds the feature cache. Subsequent epochs will be fast.")

        # --- TRAIN PHASE ---
        model.train()
        correct, total = 0, 0
        train_loss = 0.0

        for imgs, tex, labels in tqdm(train_loader, desc="Training"):
            imgs, tex, labels = imgs.to(device), tex.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, tex)
            loss = criterion(outputs, labels)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = 100 * correct / total

        # --- VALIDATION PHASE ---
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0

        with torch.no_grad():
            for imgs, tex, labels in tqdm(val_loader, desc="Validating"):
                imgs, tex, labels = imgs.to(device), tex.to(device), labels.to(device)

                outputs = model(imgs, tex)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = outputs.argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = 100 * correct / total

        # Step the learning rate scheduler based on validation accuracy
        scheduler.step(val_acc)

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # --- SAVE BEST MODEL ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_state,
                "classes": dataset.classes,
                "texture_dim": texture_dim
            }, MODEL_PATH)

            print("✓ Best model weights updated and saved!")

    print(f"\n[SUCCESS] Training Complete! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()