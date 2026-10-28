"""
dip_pipeline.py
---------------
Digital Image Processing (DIP) stages for the Plant Leaf Disease Detection
mini project. Shared by app.py (website) and train_model.py (training) so the
classifier sees exactly the same pre-processed image during training and
prediction.

Stages implemented here:
    Acquisition -> Enhancement -> Filtering -> Edge Detection ->
    Thresholding -> Morphology -> (lesion localisation for the final image)

NOTE: The classifier (Keras model) lives in app.py / train_model.py.
      Segmentation here only LOCATES suspicious regions; it never decides
      the disease name.
"""
import cv2
import numpy as np

MAX_SIDE = 1024          # images larger than this are down-scaled for speed
MODEL_SIZE = 224         # default classifier input size


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def norm(x):
    """Clip to 0-255 and convert to uint8."""
    return np.clip(x, 0, 255).astype(np.uint8)


def decode_image(data: bytes):
    """
    Decode raw image bytes into a BGR uint8 image.
    Returns (bgr_image, original_channels).
    Raises ValueError for corrupted / unreadable data.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise ValueError("The file could not be decoded as an image (it may be corrupted).")

    if img.dtype != np.uint8:                       # 16-bit PNG etc.
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    channels = 1 if img.ndim == 2 else img.shape[2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                         # composite alpha on white
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    elif img.shape[2] != 3:
        raise ValueError("Unsupported number of image channels.")
    return img, channels


def limit_size(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = max(h, w) / float(max_side)
    if scale <= 1:
        return img
    return cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Pre-processing used by the classifier (enhancement + filtering)
# --------------------------------------------------------------------------
def classifier_input_bgr(img):
    """Contrast stretching (enhancement) followed by Gaussian filtering."""
    stretched = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(stretched, (3, 3), 0)


def preprocess_for_model(img_bgr, size=MODEL_SIZE):
    """
    BGR uint8 image -> float32 RGB array (size, size, 3) with values 0..255.
    (Rescaling to the network's range happens inside the Keras model.)
    """
    processed = classifier_input_bgr(img_bgr)
    processed = cv2.resize(processed, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32)


# --------------------------------------------------------------------------
# Leaf & lesion localisation (DIP only)
# --------------------------------------------------------------------------
def segment_leaf(img):
    """Estimate the largest foreground object as a leaf mask; reject broad masks."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = img.shape[0] * img.shape[1]
    filled = np.zeros_like(mask)
    if not contours:
        return filled, False
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0.02 * total:
        return filled, False
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)

    frac = filled.mean() / 255.0
    if frac < 0.05 or frac > 0.85:
        return np.zeros_like(mask), False
    return filled, True


def find_lesions(img, leaf_mask):
    """
    Locate suspicious non-green, very dark, or whitish pixels inside the leaf.
    Colour rules and morphology produce an approximate DIP mask.
    Returns (lesion_mask uint8 0/255, leaf_pixel_count, lesion_pixel_count).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    side = max(img.shape[:2])
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, side // 100 | 1),) * 2)
    inner = cv2.erode(leaf_mask, ek)                 # ignore the leaf boundary

    inside = inner > 0
    leaf_px = int(np.count_nonzero(leaf_mask))
    if inside.sum() < 50:
        return np.zeros_like(leaf_mask), leaf_px, 0

    median_v = float(np.median(v[inside]))
    not_green = ((h < 33) | (h > 90)) & (s > 40)     # brown / yellow / purple
    dark = v < 0.45 * median_v                       # necrotic / black spots
    whitish = (s < 40) & (v > 190)                   # powdery coating
    lesion = ((not_green | dark | whitish) & inside).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, k3)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, k5)

    # remove tiny specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lesion, connectivity=8)
    min_area = max(20, int(0.001 * leaf_px))
    clean = np.zeros_like(lesion)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[lab == i] = 255
    return clean, leaf_px, int(np.count_nonzero(clean))


def annotate_final(img, label_text, diseased, lesion_mask=None, confidence=None):
    """
    Build the final annotated image.
      diseased=True  -> red translucent overlay + red boundary on lesions
      diseased=False -> leaf shown normally with a green 'Healthy Leaf' banner
      diseased=None  -> leaf shown normally with a grey 'not classified' banner
    """
    out = img.copy()
    if diseased and lesion_mask is not None and lesion_mask.any():
        red = out.copy()
        red[lesion_mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(out, 0.60, red, 0.40, 0)
        cnts, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), max(2, img.shape[1] // 300))

    if diseased:
        color = (40, 40, 200)
    elif diseased is False:
        color = (60, 140, 30)
    else:
        color = (110, 110, 110)

    text = label_text if confidence is None else f"{label_text}  |  {confidence:.1f}%"
    scale = max(0.5, img.shape[1] / 900.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    bar_h = th + base + int(20 * scale)
    bar = np.full((bar_h, out.shape[1], 3), color, np.uint8)
    cv2.putText(bar, text, (int(12 * scale), int(bar_h / 2 + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return np.vstack([out, bar])


# --------------------------------------------------------------------------
# Full DIP stage computation (Acquisition ... Morphology)
# --------------------------------------------------------------------------
def run_dip_stages(img):
    """
    Run every DIP stage on a BGR image.
    Returns dict name -> image (uint8) plus a small dict of numbers.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---- Enhancement ----
    negative = 255 - img
    contrast = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    lut = np.array([((i / 255.0) ** 1.3) * 255 for i in range(256)], dtype=np.uint8)
    gamma = cv2.LUT(img, lut)

    # ---- Filtering / sharpening ----
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(img, 5)
    lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    highboost = norm(img.astype(np.float32) + 1.5 * (img.astype(np.float32) - gaussian.astype(np.float32)))

    # ---- Edge detection ----
    rx = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, 0], [0, -1]], np.float32))
    ry = cv2.filter2D(gray, cv2.CV_32F, np.array([[0, 1], [-1, 0]], np.float32))
    roberts = norm(cv2.magnitude(rx, ry))
    px = cv2.filter2D(gray, cv2.CV_32F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32))
    py = cv2.filter2D(gray, cv2.CV_32F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32))
    prewitt = norm(cv2.magnitude(px, py))
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = norm(cv2.magnitude(sx, sy))
    laplacian_edge = lap

    # ---- Thresholding ----
    _, global_t = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    otsu_val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 5)

    # ---- Morphology (applied on the Otsu mask) ----
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(otsu, k, iterations=1)
    dilation = cv2.dilate(otsu, k, iterations=1)
    opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k)
    closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k)
    morph_clean = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, k)

    images = {
        "original": img, "gray": gray, "hsv": hsv,
        "negative": negative, "contrast": contrast, "gamma": gamma,
        "gaussian": gaussian, "median": median, "laplacian": lap, "highboost": highboost,
        "roberts": roberts, "prewitt": prewitt, "sobel": sobel, "laplacian_edge": laplacian_edge,
        "global": global_t, "otsu": otsu, "adaptive": adaptive,
        "erosion": erosion, "dilation": dilation, "opening": opening, "closing": closing,
        "morph_clean": morph_clean,
    }
    numbers = {
        "min": int(gray.min()), "max": int(gray.max()),
        "mean": round(float(gray.mean()), 2), "otsu": round(float(otsu_val), 2),
    }
    return images, numbers
