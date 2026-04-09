import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Enable HEIC/HEIF support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array. Supports JPEG, PNG, HEIC, etc."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    pil_img = Image.open(BytesIO(image_bytes))
    pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _order_corners(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = sorted(pts, key=lambda p: p[1])  # sort by y
    top = sorted(pts[:2], key=lambda p: p[0])  # top two sorted by x
    bottom = sorted(pts[2:], key=lambda p: p[0])  # bottom two sorted by x
    return [top[0], top[1], bottom[1], bottom[0]]


def _find_quad_in_edges(edges, min_area_frac=0.1):
    """Try to find a quadrilateral contour in edge image. Returns raw corner list or None."""
    sh, sw = edges.shape[:2]
    min_area = sh * sw * min_area_frac
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours[:15]:
        peri = cv2.arcLength(contour, True)
        for eps in [0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                return approx.reshape(4, 2).tolist()
    return None


def _is_reasonable_quad(corners, sw, sh):
    """Reject quads that cover nearly the whole image or are tiny."""
    area = cv2.contourArea(np.array(corners, dtype=np.float32))
    total = sw * sh
    return 0.05 * total < area < 0.96 * total


def _line_intersect(l1, l2):
    """Return intersection point of two lines, each as (x1,y1,x2,y2). Returns None if parallel."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]


def _detect_via_hough(gray):
    """Detect board via Hough line intersections — works well for frames with clear edges."""
    sh, sw = gray.shape[:2]
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 30, 100)
    min_len = min(sw, sh) * 0.20

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=min_len, maxLineGap=40)
    if lines is None or len(lines) < 4:
        return None

    lines = lines.reshape(-1, 4)
    h_lines, v_lines = [], []
    for x1, y1, x2, y2 in lines:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 25 or angle > 155:
            h_lines.append((x1, y1, x2, y2))
        elif 65 < angle < 115:
            v_lines.append((x1, y1, x2, y2))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h_lines_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
    v_lines_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

    top_line = h_lines_sorted[0]
    bot_line = h_lines_sorted[-1]
    left_line = v_lines_sorted[0]
    right_line = v_lines_sorted[-1]

    tl = _line_intersect(top_line, left_line)
    tr = _line_intersect(top_line, right_line)
    br = _line_intersect(bot_line, right_line)
    bl = _line_intersect(bot_line, left_line)

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    corners = [tl, tr, br, bl]

    # All corners must be within image bounds (allow 10% overshoot for perspective)
    for x, y in corners:
        if x < -sw * 0.10 or x > sw * 1.10 or y < -sh * 0.10 or y > sh * 1.10:
            return None

    if not _is_reasonable_quad(corners, sw, sh):
        return None

    return corners


def detect_board_corners(image_bytes: bytes) -> list:
    """Auto-detect whiteboard corners. Returns 4 points as [[x%, y%], ...] in 0-1 range.
    Order: top-left, top-right, bottom-right, bottom-left."""
    img = _decode_image(image_bytes)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Downscale for faster processing
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray.copy()
    sh, sw = small.shape[:2]

    best_corners = None

    # Strategy 1: CLAHE + Canny (best for low-contrast boards)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(small)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cand = _find_quad_in_edges(edges)
    if cand and _is_reasonable_quad(cand, sw, sh):
        best_corners = cand

    # Strategy 2: Standard Canny on original gray
    if best_corners is None:
        blurred2 = cv2.GaussianBlur(small, (5, 5), 0)
        edges2 = cv2.Canny(blurred2, 50, 150)
        edges2 = cv2.dilate(edges2, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        cand = _find_quad_in_edges(edges2)
        if cand and _is_reasonable_quad(cand, sw, sh):
            best_corners = cand

    # Strategy 3: Hough lines (excellent for boards with visible rectangular frames)
    if best_corners is None:
        best_corners = _detect_via_hough(small)

    # Strategy 4: Morphological closing to connect broken edges
    if best_corners is None:
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
        cand = _find_quad_in_edges(closed)
        if cand and _is_reasonable_quad(cand, sw, sh):
            best_corners = cand

    # Strategy 5: Brightness-based — whiteboards are typically the brightest large region
    if best_corners is None:
        _, bright = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        contours_b, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_b:
            largest = max(contours_b, key=cv2.contourArea)
            if cv2.contourArea(largest) > sh * sw * 0.1:
                hull = cv2.convexHull(largest)
                peri = cv2.arcLength(hull, True)
                for eps in [0.02, 0.05, 0.1]:
                    approx = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(approx) == 4:
                        cand = approx.reshape(4, 2).tolist()
                        if _is_reasonable_quad(cand, sw, sh):
                            best_corners = cand
                            break
                if best_corners is None:
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    cand = box.tolist()
                    if _is_reasonable_quad(cand, sw, sh):
                        best_corners = cand

    if best_corners is None:
        # Final fallback: centered rectangle at 10% margin (looks intentional)
        margin = 0.10
        best_corners = [
            [int(sw * margin), int(sh * margin)],
            [int(sw * (1 - margin)), int(sh * margin)],
            [int(sw * (1 - margin)), int(sh * (1 - margin))],
            [int(sw * margin), int(sh * (1 - margin))],
        ]

    # Scale back to original coordinates, then normalize to 0-1
    ordered = _order_corners(best_corners)
    result = []
    for x, y in ordered:
        result.append([round(x / scale / w, 4), round(y / scale / h, 4)])
    return result


def _perspective_crop(img: np.ndarray, corners: list) -> np.ndarray:
    """Apply perspective transform to crop the quadrilateral defined by corners.
    corners: [[x%, y%], ...] as fractions 0-1, order: TL, TR, BR, BL."""
    h, w = img.shape[:2]
    src_pts = np.array([[c[0] * w, c[1] * h] for c in corners], dtype=np.float32)

    # Compute output dimensions from the quadrilateral
    width_top = np.linalg.norm(src_pts[1] - src_pts[0])
    width_bot = np.linalg.norm(src_pts[2] - src_pts[3])
    out_w = int(max(width_top, width_bot))

    height_left = np.linalg.norm(src_pts[3] - src_pts[0])
    height_right = np.linalg.norm(src_pts[2] - src_pts[1])
    out_h = int(max(height_left, height_right))

    dst_pts = np.array([
        [0, 0], [out_w, 0], [out_w, out_h], [0, out_h]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def process_whiteboard(
    image_bytes: bytes,
    corners: list = None,
    sensitivity: int = 80,
    steepness: float = 0.30,
    blur_sigma: int = 51,
    denoise: bool = True,
    preserve_color: bool = False,
) -> bytes:
    """Process a whiteboard photo: remove background, return transparent PNG with ink only."""
    img = _decode_image(image_bytes)

    # Perspective crop if corners provided
    if corners:
        img = _perspective_crop(img, corners)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Step 1: Background normalization
    ksize = blur_sigma | 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0).astype(np.float64)
    blurred = np.maximum(blurred, 1.0)
    normalized = np.clip((gray / blurred) * 255.0, 0, 255).astype(np.uint8)

    # Step 2: Median filter
    filtered = cv2.medianBlur(normalized, 3)

    # Step 3: Adaptive thresholding
    block_size = max(int(min(h, w) * 0.05), 51) | 1
    c_value = max(2, int(14 - sensitivity * 0.12))

    ink_binary = cv2.adaptiveThreshold(
        filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c_value
    )

    # Step 4: Smooth alpha
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink_region = cv2.dilate(ink_binary, dilate_k)

    ink_strength = 255.0 - filtered.astype(np.float64)
    ink_pixels = ink_strength[ink_binary > 0]
    noise_floor = np.percentile(ink_pixels, 5) if len(ink_pixels) > 0 else 5.0

    alpha = 255.0 / (1.0 + np.exp(-steepness * (ink_strength - noise_floor)))
    alpha = np.clip(alpha, 0, 255)
    alpha[ink_region == 0] = 0

    # Small margin to avoid edge artifacts (only when perspective-cropped)
    if corners:
        margin = int(min(h, w) * 0.01)
        alpha[:margin, :] = 0
        alpha[-margin:, :] = 0
        alpha[:, :margin] = 0
        alpha[:, -margin:] = 0

    alpha = alpha.astype(np.uint8)

    # Step 5: Morphological cleanup
    if denoise:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

        binary = (alpha > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = h * w * 0.00004
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                alpha[labels == i] = 0

    # Step 6: Compose RGBA
    if preserve_color:
        b, g, r = cv2.split(img)
        rgba = cv2.merge([r, g, b, alpha])
    else:
        black = np.zeros((h, w), dtype=np.uint8)
        rgba = cv2.merge([black, black, black, alpha])

    pil_img = Image.fromarray(rgba, "RGBA")
    buf = BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
