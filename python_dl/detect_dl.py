import os
import torch
import torch.nn as nn
from torchvision import models

# Import the optimized preprocessing functions!
from processed import extract_image, extract_texture_features

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "model/hybrid_leaf_model.pth"
CLASS_PATH = "model/class_names.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Using Device: {device}")

# ==========================
# HYBRID ARCHITECTURE (Must match training exactly)
# ==========================
class HybridModel(nn.Module):
    def __init__(self, num_classes, texture_dim):
        super().__init__()
        
        # Base CNN (No weights needed for inference, we load our own)
        base = models.mobilenet_v2(weights=None)
        base.classifier = nn.Identity()
        self.cnn = base
        
        self.cnn_norm = nn.LayerNorm(1280)
        self.cnn_proj = nn.Linear(1280, 256)

        # Texture Branch
        self.texture_net = nn.Sequential(
            nn.Linear(texture_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Late Fusion Gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 384),
            nn.Sigmoid()
        )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, tex):
        cnn_feat = self.cnn(img)
        cnn_feat = self.cnn_norm(cnn_feat)
        cnn_feat = self.cnn_proj(cnn_feat)
        
        tex_feat = self.texture_net(tex)
        
        fused = torch.cat([cnn_feat, tex_feat], dim=1)
        gate = self.fusion_gate(fused)
        fused = fused * gate
        
        return self.classifier(fused)

# ==========================
# LOAD MODEL
# ==========================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Load the checkpoint dictionary saved during training
    # weights_only=False is required to load the list of classes correctly in older PyTorch versions
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    num_classes = len(checkpoint["classes"])
    texture_dim = checkpoint["texture_dim"]

    # Initialize model and load weights
    model = HybridModel(num_classes, texture_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval() # CRITICAL: Sets Dropout and BatchNorm to evaluation mode

    return model, checkpoint["classes"]

# ==========================
# PREDICT FUNCTION
# ==========================
def predict(image_path, model, class_names):
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return None, None

    try:
        # 1. Use the EXACT same preprocessing functions as training
        img = extract_image(image_path)
        tex = extract_texture_features(image_path)
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None, None

    # 2. Convert to Tensors and add Batch Dimension [1, Channels, H, W]
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
    tex_tensor = torch.tensor(tex, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(img_tensor, tex_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item() * 100

    return label, confidence

# ==========================
# EXECUTION
# ==========================
if __name__ == "__main__":
    print("[INFO] Loading model weights...")
    model, class_names = load_model()
    
    # Put your real-world test image path here
    test_image = "test.jpg"  

    print(f"[INFO] Running prediction on {test_image}...\n")
    label, confidence = predict(test_image, model, class_names)

    if label:
        print("="*40)
        print(f"🌿 Prediction: {label}")
        print(f"🎯 Confidence: {confidence:.2f}%")
        print("="*40 + "\n")