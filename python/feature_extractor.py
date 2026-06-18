import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


# -----------------------------
# 0. SAFE IMAGE CONVERTER
# -----------------------------
def to_uint8(image):
    if image is None:
        raise ValueError("Input image is None")

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# -----------------------------
# 1. COLOR STATISTICS
# -----------------------------
def color_stats(image):
    image = to_uint8(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
    h_std, s_std, v_std = np.std(hsv, axis=(0, 1))

    b_mean, g_mean, r_mean = np.mean(image, axis=(0, 1))
    b_std, g_std, r_std = np.std(image, axis=(0, 1))

    return [
        h_mean, s_mean, v_mean,
        h_std, s_std, v_std,
        b_mean, g_mean, r_mean,
        b_std, g_std, b_std
    ]


# -----------------------------
# 2. COLOR MOMENTS (NEW 🔥)
# -----------------------------
def color_moments(image):
    image = to_uint8(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    moments = []

    for i in range(3):
        channel = hsv[:, :, i].astype(np.float32)

        mean = np.mean(channel)
        std = np.std(channel)

        skewness = np.mean(((channel - mean) / (std + 1e-6)) ** 3)
        kurtosis = np.mean(((channel - mean) / (std + 1e-6)) ** 4)

        moments.extend([mean, std, skewness, kurtosis])

    return moments


# -----------------------------
# 3. HSV COLOR HISTOGRAM
# -----------------------------
def hsv_histogram(image, bins=(8, 8, 8)):
    image = to_uint8(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()

    return hist.astype(np.float32)


# -----------------------------
# 4. AUTO LEAF SEGMENTATION (NEW 🔥)
# -----------------------------
def get_leaf_mask(image):
    """
    Removes background noise (VERY important improvement)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# -----------------------------
# 5. SHAPE FEATURES
# -----------------------------
def shape_features(mask):
    area = np.sum(mask > 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [area, 0, 0, 0]

    cnt = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / (h + 1e-6)
    contour_area = cv2.contourArea(cnt)

    return [area, perimeter, aspect_ratio, contour_area]


# -----------------------------
# 6. DISEASE AREA %
# -----------------------------
def disease_area_feature(mask):
    total = mask.size
    diseased = np.sum(mask > 0)

    return [diseased / (total + 1e-6)]


# -----------------------------
# 7. EDGE DENSITY (NEW 🔥)
# -----------------------------
def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return [np.sum(edges > 0) / (edges.size + 1e-6)]


# -----------------------------
# 8. LBP TEXTURE
# -----------------------------
def lbp_features(gray):
    radius = 3
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    return hist.tolist()


# -----------------------------
# 9. GLCM TEXTURE
# -----------------------------
def glcm_features(gray):
    glcm = graycomatrix(
        gray,
        distances=[1, 2],
        angles=[0, np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    return [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'dissimilarity').mean()
    ]


# -----------------------------
# 10. FULL PIPELINE
# -----------------------------
def extract_features(image, mask=None):
    image = to_uint8(image)

    if mask is None:
        mask = get_leaf_mask(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    # CORE COLOR FEATURES
    features.extend(color_stats(image))
    features.extend(color_moments(image))

    # DISTRIBUTION FEATURES
    features.extend(hsv_histogram(image))

    # STRUCTURE FEATURES
    features.extend(shape_features(mask))
    features.extend(disease_area_feature(mask))

    # TEXTURE FEATURES
    features.extend(lbp_features(gray))
    features.extend(glcm_features(gray))

    # EDGE FEATURES
    features.extend(edge_density(gray))

    return np.array(features, dtype=np.float32)