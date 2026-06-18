import cv2
import numpy as np

# -----------------------------
# 1. Resize Image
# -----------------------------
def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)


# -----------------------------
# 2. CLAHE (Contrast Enhancement)
# -----------------------------
def apply_clahe(image):
    """
    Enhances leaf disease visibility using LAB color space
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


# -----------------------------
# 3. Noise Reduction
# -----------------------------
def denoise(image):
    """
    Reduces small noise while preserving edges
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


# -----------------------------
# 4. Leaf Segmentation (HSV Mask)
# -----------------------------
def segment_leaf(image):
    """
    Extracts green leaf region and removes most background
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # apply mask
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask


# -----------------------------
# 5. Morphological Processing (NEW)
# -----------------------------
def morphological_processing(image, mask=None):
    """
    Cleans segmentation noise and strengthens leaf structure
    """
    kernel = np.ones((5, 5), np.uint8)

    # Close small holes
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return opened


# -----------------------------
# 6. Full Preprocessing Pipeline
# -----------------------------
def preprocess_image(image_path, return_mask=False):
    """
    Complete preprocessing pipeline for:
    - training
    - inference
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    # Step 1: Resize
    image = resize_image(image)

    # Step 2: CLAHE
    image = apply_clahe(image)

    # Step 3: Denoise
    image = denoise(image)

    # Step 4: Segmentation
    segmented, mask = segment_leaf(image)

    # Step 5: Morphology
    processed = morphological_processing(segmented, mask)

    # Step 6: Normalize (for feature extraction consistency)
    processed = processed.astype(np.float32) / 255.0

    if return_mask:
        return processed, mask

    return processed