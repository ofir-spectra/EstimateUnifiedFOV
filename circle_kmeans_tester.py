
import cv2
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread('c85_2201_00_0697-00-org.png')
if img is None:
    raise FileNotFoundError('Image not found!')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def nothing(x):
    pass

screen_width = 1280
screen_height = 720
cv2.namedWindow('Binary Mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Binary Mask', screen_width, screen_height)
cv2.createTrackbar('Mode', 'Binary Mask', 0, 2, nothing)  # 0:RGB, 1:Adaptive, 2:Otsu
cv2.createTrackbar('R thresh', 'Binary Mask', 128, 255, nothing)
cv2.createTrackbar('G thresh', 'Binary Mask', 128, 255, nothing)
cv2.createTrackbar('B thresh', 'Binary Mask', 128, 255, nothing)

cv2.createTrackbar('Block size', 'Binary Mask', 11, 50, nothing)  # for adon, -org.pngaptive
cv2.createTrackbar('C', 'Binary Mask', 2, 20, nothing)  # for adaptive
# HoughCircles parameter sliders
cv2.createTrackbar('minDist', 'Binary Mask', 100, 500, nothing)
cv2.createTrackbar('param2', 'Binary Mask', 15, 100, nothing)
cv2.createTrackbar('minRadius', 'Binary Mask', 20, 200, nothing)
cv2.createTrackbar('maxRadius', 'Binary Mask', 200, 600, nothing)

while True:
    mode = cv2.getTrackbarPos('Mode', 'Binary Mask')
    r_thr = cv2.getTrackbarPos('R thresh', 'Binary Mask')
    g_thr = cv2.getTrackbarPos('G thresh', 'Binary Mask')
    b_thr = cv2.getTrackbarPos('B thresh', 'Binary Mask')
    block_size = cv2.getTrackbarPos('Block size', 'Binary Mask')
    c_val = cv2.getTrackbarPos('C', 'Binary Mask')
    minDist = cv2.getTrackbarPos('minDist', 'Binary Mask')
    param2 = cv2.getTrackbarPos('param2', 'Binary Mask')
    minRadius = cv2.getTrackbarPos('minRadius', 'Binary Mask')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Binary Mask')
    if block_size % 2 == 0:
        block_size += 1
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if mode == 0:
        # RGB threshold mode
        masks = []
        for i, thr in enumerate([r_thr, g_thr, b_thr]):
            channel = img_rgb[:, :, i]
            _, mask = cv2.threshold(channel, thr, 255, cv2.THRESH_BINARY)
            masks.append(mask)
        binary = cv2.bitwise_and(masks[0], masks[1])
        binary = cv2.bitwise_and(binary, masks[2])
    elif mode == 1:
        # Adaptive threshold mode
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, c_val)
    else:
        # Otsu's threshold mode
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find nonzero points (likely circle regions)
    points = np.column_stack(np.where(binary > 0))
    # Resize for display
    scale = min(screen_width / img_rgb.shape[1], screen_height / img_rgb.shape[0], 1.0)
    disp_img = cv2.resize(img_rgb, (int(img_rgb.shape[1]*scale), int(img_rgb.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    display = disp_img.copy()
    # Draw only on the original (resized) image
    # Contour detection and fit circle to all large contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500  # You may need to adjust this value
    # Find the 4 largest contours and fit a circle to each
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Morphological opening to separate blobs
    kernel = np.ones((15, 15), np.uint8)
    binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Connected components to find each binary circle
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_opened)
    overlay_colors = [(0,255,255), (0,255,0), (255,0,255), (255,255,0)]
    # Filter out very small components, then take the 4 largest
    min_area = 50
    component_areas = [(label, stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
    large_components = [label for label, area in component_areas if area >= min_area]
    # Sort by area, descending, and take up to 4
    large_components = sorted(large_components, key=lambda l: stats[l, cv2.CC_STAT_AREA], reverse=True)[:4]
    for i, label in enumerate(large_components):
        mask = (labels == label).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
        # maxLoc is (x, y) in (col, row) order in the mask's coordinate system
        x_mask, y_mask = maxLoc
        r_mask = int(maxVal)
        # Scale coordinates to display image
        y_scale = display.shape[0] / mask.shape[0]
        x_scale = display.shape[1] / mask.shape[1]
        x_disp = int(x_mask * x_scale)
        y_disp = int(y_mask * y_scale)
        r_disp = int(r_mask * (x_scale + y_scale) / 2)
        color = overlay_colors[i % len(overlay_colors)]
        print(f"Circle {i+1}: x={x_disp}, y={y_disp}, r={r_disp}")
        cv2.circle(display, (x_disp, y_disp), r_disp, color, 2)
        cv2.circle(display, (x_disp, y_disp), 4, (0, 0, 255), -1)
        cv2.putText(display, f"r={r_disp}", (x_disp - 20, y_disp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- Overlap visualization for circles 2, 3, 4 ---
    if len(large_components) >= 4:
        # Get centers and radii for circles 2, 3, 4
        centers_radii = []
        for idx in range(1, 4):
            label = large_components[idx]
            mask = (labels == label).astype(np.uint8)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
            x, y = maxLoc
            r = int(maxVal)
            centers_radii.append((x, y, r, mask))

        max_r = max(r for (_, _, r, _) in centers_radii)
        out_h = int(2 * max_r)
        out_w = int(1.5 * max_r)
        overlap_view = np.zeros((out_h, out_w), dtype=np.uint8)

        for (x, y, r, mask) in centers_radii:
            # For each pixel in the mask, if it's 1, map to overlap_view
            ys, xs = np.where(mask > 0)
            # Compute offset to center this mask in overlap_view
            center_y = out_h // 2
            center_x = out_w // 2
            dy = ys - y
            dx = xs - x
            oy = center_y + dy
            ox = center_x + dx
            # Only keep points inside bounds
            valid = (oy >= 0) & (oy < out_h) & (ox >= 0) & (ox < out_w)
            overlap_view[oy[valid], ox[valid]] += 1

        # Colorize the overlap_view for display (0=black, 1=blue, 2=green, 3=red)
        color_map = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        color_map[overlap_view == 1] = (255, 0, 0)   # Blue
        color_map[overlap_view == 2] = (0, 255, 0)   # Green
        color_map[overlap_view == 3] = (0, 0, 255)   # Red
        # Resize for display in the top right corner
        thumb_h = display.shape[0] // 3
        thumb_w = display.shape[1] // 4
        color_map_disp = cv2.resize(color_map, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        # Place in top right of display
        display[0:thumb_h, -thumb_w:] = color_map_disp
    mask_bgr = cv2.cvtColor(binary_opened, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.resize(mask_bgr, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_AREA)
    combined = np.hstack([display, mask_bgr])
    cv2.imshow('Binary Mask', combined)
    key = cv2.waitKey(50)
    if key == 27:  # ESC to exit
        break
cv2.destroyAllWindows()
