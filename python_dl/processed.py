import cv2
import numpy as np
from skimage.feature import local_binary_pattern

IMAGE_SIZE = 224

# Standard ImageNet normalization values required by MobileNetV2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def extract_image(path):
    """
    Extracts and standardizes the raw image for the CNN branch.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found: {path}")

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Scale to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Apply ImageNet normalization (CRITICAL for MobileNetV2)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # Convert to PyTorch format: (Channels, Height, Width)
    img = np.transpose(img, (2, 0, 1))
    return img


def extract_texture_features(path):
    """
    Extracts handcrafted LBP and HSV histograms for the MLP branch.
    Utilizes dynamic thresholding to preserve disease lesions.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found: {path}")

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # 1. Robust Segmentation via Saturation Channel
    # Leaves (even diseased) are generally more saturated than backgrounds.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    
    # Otsu's thresholding dynamically finds the best separation point between foreground/background
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill small holes in the leaf
    # --- ADD THESE 3 LINES FOR DEBUGGING ---
    segmented_debug = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("debug_mask.jpg", mask)
    cv2.imwrite("debug_segmented.jpg", segmented_debug)
    # ---------------------------------------
    # 2. Local Binary Pattern (LBP) Extraction
    # We run this on grayscale, but only count pixels inside our leaf mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    # Extract ONLY the pixels that belong to the leaf (mask > 0)
    # If we used the whole image, the empty background would heavily skew the histogram
    valid_lbp_pixels = lbp[mask > 0]
    
    # Fallback in the rare case Otsu completely fails and the mask is empty
    if len(valid_lbp_pixels) == 0:
        valid_lbp_pixels = lbp.ravel()

    lbp_hist, _ = np.histogram(
        valid_lbp_pixels,
        bins=np.arange(0, 11),
        range=(0, 10),
        density=True
    )

    # 3. HSV Color Histogram
    # OpenCV's calcHist accepts a mask, automatically ignoring the background
    hsv_hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        mask,  # <--- Only calculates colors within the leaf
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )

    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # 4. Feature Fusion & Normalization
    features = np.concatenate([lbp_hist, hsv_hist]).astype(np.float32)

    # L2 normalize the entire combined vector (stable for the fusion gate in PyTorch)
    features = features / (np.linalg.norm(features) + 1e-6)

    return features