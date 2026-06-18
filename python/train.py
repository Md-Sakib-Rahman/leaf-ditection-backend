import os
import json
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from preprocess import preprocess_image
from feature_extractor import extract_features


# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset/PlantVillage"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2


# -----------------------------
# LOAD DATASET
# -----------------------------
X, y = [], []

class_names = sorted(os.listdir(DATASET_PATH))
print(f"Found {len(class_names)} classes\n")

for label, class_name in enumerate(class_names):
    class_path = os.path.join(DATASET_PATH, class_name)
    images = os.listdir(class_path)

    print(f"Processing {class_name} ({len(images)} images)")

    for img_name in images:
        img_path = os.path.join(class_path, img_name)

        try:
            img = preprocess_image(img_path)
            features = extract_features(img)

            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"Skip {img_path}: {e}")


X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nDataset loaded:")
print("Samples:", len(X))
print("Feature shape:", X.shape)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)


# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# SMOTE (BALANCE CLASSES)
# -----------------------------
print("\nApplying SMOTE...")
smote = SMOTE(random_state=SEED)
X_train, y_train = smote.fit_resample(X_train, y_train)


# -----------------------------
# PCA (DIMENSION REDUCTION)
# -----------------------------
print("Applying PCA...")

pca = PCA(n_components=0.95, random_state=SEED)  # keep 95% variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# -----------------------------
# MODELS
# -----------------------------
svm_base = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

models = {
    "Calibrated_SVM": CalibratedClassifierCV(
        estimator=svm_base,
        method="sigmoid",
        cv=3
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED
    ),

    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist"
    )
}


best_model = None
best_name = ""
best_acc = 0


# -----------------------------
# TRAINING LOOP
# -----------------------------
print("\nTraining models...\n")

for name, model in models.items():
    print(f"\n================ {name} ================")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"\n{name} Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds))

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name


# -----------------------------
# SAVE ARTIFACTS
# -----------------------------
print("\nSaving best model...")

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))

with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
    json.dump({i: name for i, name in enumerate(class_names)}, f)


# -----------------------------
# FINAL RESULT
# -----------------------------
print("\n===================================")
print(f"BEST MODEL: {best_name}")
print(f"BEST ACCURACY: {best_acc * 100:.2f}%")
print("Saved scaler + PCA + model")
print("===================================")