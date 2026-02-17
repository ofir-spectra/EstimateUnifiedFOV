# IMPORTANT: All code changes must be thoroughly checked, run, and validated for correctness and robustness before presenting to the user. Never provide untested or incomplete solutions.

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QPalette, QColor, QImage, QPainter, QPen, QBrush
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def analyze_and_overlay(self, img, rgb_threshold, gl_radius_multiplier=0.5):
        import cv2
        import numpy as np
        # --- Mask extraction ---
        r_thr = g_thr = b_thr = rgb_threshold
        masks = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, thr in enumerate([r_thr, g_thr, b_thr]):
            channel = img_rgb[:, :, i]
            _, mask = cv2.threshold(channel, thr, 255, cv2.THRESH_BINARY)
            masks.append(mask)
        binary = cv2.bitwise_and(masks[0], masks[1])
        binary = cv2.bitwise_and(binary, masks[2])
        kernel = np.ones((15, 15), np.uint8)
        binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_opened)
        overlay = img.copy()
        # Fixed colors for each spatial location: [top-left, top-right, bottom-left, bottom-right]
        overlay_colors = [
            (0, 0, 255, 128),     # Blue (top-left)
            (0, 255, 255, 128),   # Yellow (top-right)
            (255, 0, 255, 128),   # Magenta (bottom-left)
            (0, 255, 0, 128),     # Green (bottom-right)
        ]
        min_area = 50
        component_areas = [(label, stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
        # Only keep the 4 largest components by area (ignore extras)
        large_components = [label for label, area in sorted(component_areas, key=lambda x: x[1], reverse=True)[:4] if area >= min_area]
        centers_radii = []
        for i, label in enumerate(large_components):
            mask = (labels == label).astype(np.uint8)
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                points = np.column_stack((xs, ys)).astype(np.float32)
                (circ_x, circ_y), radius = cv2.minEnclosingCircle(points)
                r = int(round(radius))
            else:
                circ_x, circ_y, r = cx, cy, 0
            centers_radii.append((cx, cy, r, mask))
        # Assign fixed colors by spatial location (top-left, top-right, bottom-left, bottom-right)
        if len(centers_radii) == 4:
            sorted_crs = sorted(centers_radii, key=lambda c: (c[1], c[0]))  # sort by y, then x
            # Top two (lowest y), then left/right
            top_two = sorted(sorted_crs[:2], key=lambda c: c[0])
            bottom_two = sorted(sorted_crs[2:], key=lambda c: c[0])
            ordered_crs = [top_two[0], top_two[1], bottom_two[0], bottom_two[1]]  # TL, TR, BL, BR
            for i, (cx, cy, r, mask) in enumerate(ordered_crs):
                color = overlay_colors[i]
                overlay = self.draw_transparent_circle(overlay, (int(cx), int(cy)), int(r), color)
                overlay = self.draw_transparent_circle(overlay, (int(cx), int(cy)), 4, (255, 255, 255, 180), fill=True)
                font_scale = 1.8
                thickness = 5
                yellow = (0, 255, 255)  # BGR for yellow
                cv2.putText(overlay, f"r={int(r)}", (int(cx) - 40, int(cy) - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, yellow, thickness)
                cv2.putText(overlay, f"({int(cx)},{int(cy)})", (int(cx) - 90, int(cy) + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, yellow, thickness)
            centers_radii = ordered_crs
        else:
            for i, (cx, cy, r, mask) in enumerate(centers_radii):
                color = overlay_colors[i % len(overlay_colors)]
                overlay = self.draw_transparent_circle(overlay, (int(cx), int(cy)), int(r), color)
                overlay = self.draw_transparent_circle(overlay, (int(cx), int(cy)), 4, (255, 255, 255, 180), fill=True)
                font_scale = 1.8
                thickness = 5
                yellow = (0, 255, 255)  # BGR for yellow
                cv2.putText(overlay, f"r={int(r)}", (int(cx) - 40, int(cy) - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, yellow, thickness)
                cv2.putText(overlay, f"({int(cx)},{int(cy)})", (int(cx) - 90, int(cy) + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, yellow, thickness)
        # Overlap and internal ellipsoid (data-driven: fit to union of masks 2,3,4)
        overlap_internal_radius = None
        ellipsoid_d1 = None
        ellipsoid_d2 = None
        if len(centers_radii) >= 4:
            crs = centers_radii[1:4]
            # All masks are in original image coordinates
            # Compute union of the three masks
            mask_union = np.zeros_like(crs[0][3], dtype=np.uint8)
            for (_, _, _, mask) in crs:
                mask_union = cv2.bitwise_or(mask_union, mask)
            # Fit ellipse to the union region (if enough points)
            ys, xs = np.where(mask_union > 0)
            pts = np.column_stack((xs, ys)).astype(np.int32)
            if len(pts) >= 5:
                ellipse = cv2.fitEllipse(pts)
                (ex, ey), (MA, ma), angle = ellipse
                ellipsoid_d1 = round(MA, 1)
                ellipsoid_d2 = round(ma, 1)
                overlap_internal_radius = None
            else:
                ellipsoid_d1 = None
                ellipsoid_d2 = None
        # Build right-side overlay image (overlap/union) with centered, colored, and alpha-blended masks
        overlap_img = None
        if len(centers_radii) >= 4:
            crs = centers_radii[1:4]
            max_r = max(r for (_, _, r, _) in crs)
            out_h = int(2 * max_r)
            out_w = int(2 * max_r)
            center_y = out_h // 2
            center_x = out_w // 2
            # Output canvas for blending
            out_img = np.zeros((out_h, out_w, 3), dtype=np.float32)
            out_alpha = np.zeros((out_h, out_w), dtype=np.float32)
            # Use the exact same overlay_colors and alpha as left overlay (for circles 2,3,4)
            # Make right image alpha blending more transparent (match left, e.g., 0.2)
            # Use same fixed colors for right image, with lower alpha
            overlay_colors_overlap = [
                (0, 0, 255, 0.1),     # Blue (top-left)
                (0, 255, 255, 0.1),   # Yellow (top-right)
                (255, 0, 255, 0.1),   # Magenta (bottom-left)
                (0, 255, 0, 0.1),     # Green (bottom-right)
            ]
            # Use same spatial order as left
            if len(crs) == 4:
                sorted_crs = sorted(crs, key=lambda c: (c[1], c[0]))
                top_two = sorted(sorted_crs[:2], key=lambda c: c[0])
                bottom_two = sorted(sorted_crs[2:], key=lambda c: c[0])
                ordered_crs = [top_two[0], top_two[1], bottom_two[0], bottom_two[1]]
            else:
                ordered_crs = crs
            for idx, (x, y, r, mask) in enumerate(ordered_crs):
                color = overlay_colors_overlap[idx % len(overlay_colors_overlap)]
                ys, xs = np.where(mask > 0)
                dy = ys - y
                dx = xs - x
                oy = center_y + dy
                ox = center_x + dx
                valid = (oy >= 0) & (oy < out_h) & (ox >= 0) & (ox < out_w)
                oy = oy[valid]
                ox = ox[valid]
                # Create colored mask (multiply mask by RGB color)
                src_rgb = np.array(color[:3], dtype=np.float32)
                src_alpha = color[3]
                src_img = np.tile(src_rgb, (len(oy), 1))
                dst_rgb = out_img[oy, ox]
                dst_alpha = out_alpha[oy, ox]
                out_a = src_alpha + dst_alpha * (1 - src_alpha)
                dst_alpha_exp = dst_alpha[:, None]
                out_a_exp = out_a[:, None]
                out_rgb = (src_img * src_alpha + dst_rgb * dst_alpha_exp * (1 - src_alpha)) / np.clip(out_a_exp, 1e-6, 1)
                out_img[oy, ox] = out_rgb
                out_alpha[oy, ox] = out_a
            rgb_map = np.clip(out_img, 0, 255).astype(np.uint8)
            # Fit ellipsoid to the largest external contour of the union of the three centered masks
            mask_union = np.zeros((out_h, out_w), dtype=np.uint8)
            mask_intersection = np.ones((out_h, out_w), dtype=np.uint8) * 255
            for (x, y, r, mask) in crs:
                ys, xs = np.where(mask > 0)
                dy = ys - y
                dx = xs - x
                oy = center_y + dy
                ox = center_x + dx
                valid = (oy >= 0) & (oy < out_h) & (ox >= 0) & (ox < out_w)
                oy = oy[valid]
                ox = ox[valid]
                mask_union[oy, ox] = 255  # Use 255 for binary mask
                mask_temp = np.zeros((out_h, out_w), dtype=np.uint8)
                mask_temp[oy, ox] = 255
                mask_intersection = cv2.bitwise_and(mask_intersection, mask_temp)
            # Data-driven effective ellipse estimation
            # 1. Find center of mass of the mask_union
            M = cv2.moments(mask_union)
            if M["m00"] == 0:
                cx, cy = mask_union.shape[1] // 2, mask_union.shape[0] // 2
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            # === BEGIN: Data-driven ellipse with center shift for containment ===
            erode_kernel = np.ones((11, 11), np.uint8)
            mask_eroded = cv2.erode(mask_union, erode_kernel, iterations=1)
            # 1. Initial center of mass
            M = cv2.moments(mask_eroded)
            if M["m00"] == 0:
                cx, cy = mask_eroded.shape[1] // 2, mask_eroded.shape[0] // 2
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            # 2. Find left/right extents and their offsets
            def get_lr_extents(mask, cx, cy):
                max_left = 0
                max_right = 0
                y_best = cy
                for y in range(mask.shape[0]):
                    row = mask[y, :]
                    if row[cx] == 0:
                        continue
                    left = 0
                    for x in range(cx, -1, -1):
                        if row[x] == 0:
                            break
                        left += 1
                    right = 0
                    for x in range(cx, mask.shape[1]):
                        if row[x] == 0:
                            break
                        right += 1
                    if left + right > max_left + max_right:
                        max_left = left
                        max_right = right
                        y_best = y
                return max_left, max_right, y_best

            # 3. Find top/bottom extents and their offsets
            def get_tb_extents(mask, cx, cy):
                max_top = 0
                max_bottom = 0
                x_best = cx
                for x in range(mask.shape[1]):
                    col = mask[:, x]
                    if col[cy] == 0:
                        continue
                    top = 0
                    for y in range(cy, -1, -1):
                        if col[y] == 0:
                            break
                        top += 1
                    bottom = 0
                    for y in range(cy, mask.shape[0]):
                        if col[y] == 0:
                            break
                        bottom += 1
                    if top + bottom > max_top + max_bottom:
                        max_top = top
                        max_bottom = bottom
                        x_best = x
                return max_top, max_bottom, x_best

            # 4. Iteratively adjust center to equalize left/right and top/bottom
            # Horizontal (left/right)
            max_iter = 10
            for _ in range(max_iter):
                left, right, y_best = get_lr_extents(mask_eroded, cx, cy)
                diff_lr = right - left
                if abs(diff_lr) <= 1:
                    break
                cx += int(np.sign(diff_lr) * abs(diff_lr) // 2)
                cx = np.clip(cx, 0, mask_eroded.shape[1]-1)
            # Vertical (top/bottom)
            for _ in range(max_iter):
                top, bottom, x_best = get_tb_extents(mask_eroded, cx, cy)
                diff_tb = bottom - top
                if abs(diff_tb) <= 1:
                    break
                cy += int(np.sign(diff_tb) * abs(diff_tb) // 2)
                cy = np.clip(cy, 0, mask_eroded.shape[0]-1)

            # --- Improved: Find maximal extents and orientation from mask_intersection ---
            # Find all nonzero points in mask_intersection
            pts = np.column_stack(np.where(mask_intersection > 0))
            if len(pts) < 2:
                # Fallback to previous logic if not enough points
                left, right, _ = get_lr_extents(mask_eroded, cx, cy)
                top, bottom, _ = get_tb_extents(mask_eroded, cx, cy)
                a = int(min(left, right) * 2)
                b = int(min(top, bottom) * 2)
                angle = 0.0
                center = (int(cx), int(cy))
                axes = (a // 2, b // 2)
                d1 = a
                d2 = b
                hor_left = hor_right = vert_top = vert_bottom = None
            else:
                # For each column, find topmost and bottommost points
                min_y = np.min(pts[:,0])
                max_y = np.max(pts[:,0])
                min_x = np.min(pts[:,1])
                max_x = np.max(pts[:,1])
                # Vertical (minor axis): find the two points with maximal vertical distance
                vert_top = None
                vert_bottom = None
                max_vert_dist = 0
                vert_x = None
                for x in range(min_x, max_x+1):
                    col_pts = pts[pts[:,1]==x]
                    if len(col_pts) == 0:
                        continue
                    y_top = np.min(col_pts[:,0])
                    y_bottom = np.max(col_pts[:,0])
                    dist = y_bottom - y_top
                    if dist > max_vert_dist:
                        max_vert_dist = dist
                        vert_top = (x, y_top)
                        vert_bottom = (x, y_bottom)
                        vert_x = x
                # Horizontal (major axis): find the two points with maximal horizontal distance
                hor_left = None
                hor_right = None
                max_hor_dist = 0
                hor_y = None
                for y in range(min_y, max_y+1):
                    row_pts = pts[pts[:,0]==y]
                    if len(row_pts) == 0:
                        continue
                    x_left = np.min(row_pts[:,1])
                    x_right = np.max(row_pts[:,1])
                    dist = x_right - x_left
                    if dist > max_hor_dist:
                        max_hor_dist = dist
                        hor_left = (x_left, y)
                        hor_right = (x_right, y)
                        hor_y = y

                # Center: adjust so that distances from center to top and bottom (and left/right) are equal
                if vert_top and vert_bottom:
                    # Initial guess: midpoint
                    cx = vert_x
                    cy = (vert_top[1] + vert_bottom[1]) // 2
                    # Calculate offsets to equalize top/bottom
                    top_dist = cy - vert_top[1]
                    bottom_dist = vert_bottom[1] - cy
                    offset = (top_dist - bottom_dist) // 2
                    cy -= offset
                    # Recompute distances
                    top_dist = cy - vert_top[1]
                    bottom_dist = vert_bottom[1] - cy
                    d2 = 2 * min(top_dist, bottom_dist)
                else:
                    cx, cy = int(np.mean(pts[:,1])), int(np.mean(pts[:,0]))
                    d2 = max_vert_dist

                if hor_left and hor_right:
                    cy_h = hor_y
                    cx = (hor_left[0] + hor_right[0]) // 2
                    # Calculate offsets to equalize left/right
                    left_dist = cx - hor_left[0]
                    right_dist = hor_right[0] - cx
                    offset = (left_dist - right_dist) // 2
                    cx -= offset
                    # Recompute distances
                    left_dist = cx - hor_left[0]
                    right_dist = hor_right[0] - cx
                    d1 = 2 * min(left_dist, right_dist)
                else:
                    d1 = max_hor_dist

                center = (int(cx), int(cy))
                # Angle: orientation of major axis (horizontal)
                if hor_left and hor_right:
                    dx = hor_right[0] - hor_left[0]
                    dy = hor_right[1] - hor_left[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                else:
                    angle = 0.0
                axes = (int(d1 // 2), int(d2 // 2))
                # Ensure axes do not exceed mask extents
                axes = (min(axes[0], (max_x-min_x)//2), min(axes[1], (max_y-min_y)//2))
                d1 = axes[0]*2
                d2 = axes[1]*2

            # --- Overlay: keep color, brighten intersection only (right image only) ---
            overlay_right = rgb_map.copy()
            white_mask = np.zeros_like(rgb_map, dtype=np.uint8)
            white_mask[mask_intersection > 0] = (255,255,255)
            alpha_inter = 0.45
            cv2.addWeighted(white_mask, alpha_inter, overlay_right, 1 - alpha_inter, 0, overlay_right)
            rgb_map[:] = overlay_right

            # Draw the ellipse
            cv2.ellipse(rgb_map, center, axes, angle, 0, 360, (255,255,255), 2)
            # Draw major and minor axes centered at ellipse center and oriented
            angle_rad = np.radians(angle)
            # Major axis (d1)
            dx1 = int(np.cos(angle_rad) * axes[0])
            dy1 = int(np.sin(angle_rad) * axes[0])
            pt1 = (center[0] - dx1, center[1] - dy1)
            pt2 = (center[0] + dx1, center[1] + dy1)
            cv2.line(rgb_map, pt1, pt2, (255,255,255), 2)
            # Minor axis (d2), perpendicular to major
            dx2 = int(-np.sin(angle_rad) * axes[1])
            dy2 = int(np.cos(angle_rad) * axes[1])
            pt3 = (center[0] - dx2, center[1] - dy2)
            pt4 = (center[0] + dx2, center[1] + dy2)
            cv2.line(rgb_map, pt3, pt4, (255,255,255), 2)
            # Annotate d1, d2 in white
            cv2.putText(rgb_map, f"d1={d1:.1f}", (center[0]+10, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(rgb_map, f"d2={d2:.1f}", (center[0]+10, center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # === END: Improved data-driven ellipse with orientation ===
            overlap_img = rgb_map
        # Calculate % usage of sensor
        percent_usage = None
        if len(centers_radii) == 4:
            mask_union = np.zeros_like(centers_radii[0][3], dtype=np.uint8)
            for (_, _, _, mask) in centers_radii:
                mask_union = cv2.bitwise_or(mask_union, mask)
            percent_usage = 100.0 * np.sum(mask_union > 0) / (mask_union.shape[0] * mask_union.shape[1])
            percent_usage = round(percent_usage, 1)
        
        # Calculate average gray level for each circle
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_gray_levels = []
        for i, (cx, cy, r, mask) in enumerate(centers_radii):
            if r > 0:
                # Create elliptical mask with specified radius multiplier for central pixels
                central_mask = np.zeros_like(img_gray, dtype=np.uint8)
                # Use radius multiplier (e.g., r/2, r/4, or 1*r)
                sampling_radius = int(r * gl_radius_multiplier)
                cv2.circle(central_mask, (int(cx), int(cy)), sampling_radius, 255, -1)
                # Calculate average only for central pixels
                central_pixels = img_gray[central_mask > 0]
                if len(central_pixels) > 0:
                    avg_gl = np.mean(central_pixels)
                    avg_gray_levels.append(round(avg_gl, 1))
                    
                    # Add semi-transparent overlay showing sampled region for Avg. GL
                    # Use white color with low alpha for visibility
                    overlay = self.draw_transparent_circle(overlay, (int(cx), int(cy)), sampling_radius, (255, 255, 255, 60))
                    
                    # Add text overlay on the image showing "Avg. GL = XXX"
                    # Adjust spacing to match the distance between r= and (cx,cy) lines
                    text = f"Avg.GL={avg_gl:.1f}"
                    text_y = int(cy) + 160  # Changed from +90 to +160 for better spacing
                    cv2.putText(overlay, text, (int(cx) - 90, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 5)
                else:
                    avg_gray_levels.append(None)
            else:
                avg_gray_levels.append(None)
        
        radii = [r for (_, _, r, _) in centers_radii]
        return overlay, overlap_img, radii, overlap_internal_radius, d1, d2, percent_usage, avg_gray_levels
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Circle Analyzer")
        self.setGeometry(100, 100, 1000, 800)
        self.set_dark_theme()
        
        # Settings file path
        import os
        self.settings_file = os.path.join(os.path.expanduser("~"), ".circle_analyzer_settings.json")
        
        self.init_ui()
        
        # Load saved settings after UI is initialized
        self.load_settings()

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)

    def init_ui(self):
        from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QScrollArea, QSlider, QLabel as QtLabel, QProgressBar, QComboBox, QDoubleSpinBox
        import os
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # --- Top bar widget for controls ---
        self.top_bar = QWidget()
        self.top_bar_layout = QHBoxLayout()
        self.top_bar.setLayout(self.top_bar_layout)

        # Buttons
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        self.top_bar_layout.addWidget(self.open_button)
        self.folder_button = QPushButton("Process Folder")
        self.folder_button.clicked.connect(self.process_folder_safe)
        self.top_bar_layout.addWidget(self.folder_button)
        self.analyze_data_button = QPushButton("Analyze Data")
        self.analyze_data_button.clicked.connect(self.analyze_csv_data)
        self.top_bar_layout.addWidget(self.analyze_data_button)

        # Slider and label
        self.threshold_label = QtLabel("RGB Threshold: 1")
        self.top_bar_layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(16)
        self.threshold_slider.setTickInterval(1)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.top_bar_layout.addWidget(self.threshold_slider)

        # Radius multiplier dropdown for Avg. GL calculation
        self.radius_label = QtLabel("Avg GL Radius:")
        self.top_bar_layout.addWidget(self.radius_label)
        self.radius_combo = QComboBox()
        self.radius_combo.addItem("1/4 × r", 0.25)
        self.radius_combo.addItem("1/2 × r", 0.5)
        self.radius_combo.addItem("1 × r", 1.0)
        self.radius_combo.setCurrentIndex(1)  # Default to 1/2 × r
        self.radius_combo.currentIndexChanged.connect(self.on_radius_changed)
        self.top_bar_layout.addWidget(self.radius_combo)

        # STD Requirement spinbox for CSV analysis
        self.std_label = QtLabel("STD Req.:")
        self.top_bar_layout.addWidget(self.std_label)
        self.std_spinbox = QDoubleSpinBox()
        self.std_spinbox.setMinimum(0.1)
        self.std_spinbox.setMaximum(10.0)
        self.std_spinbox.setValue(1.0)  # Default 1%
        self.std_spinbox.setSingleStep(0.1)
        self.std_spinbox.setSuffix(" %")
        self.std_spinbox.setMaximumWidth(80)
        self.std_spinbox.valueChanged.connect(self.on_std_changed)  # Save when STD changes
        self.top_bar_layout.addWidget(self.std_spinbox)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate by default
        self.progress_bar.setVisible(False)
        self.top_bar_layout.addWidget(self.progress_bar)

        # Status label for showing current file being processed
        self.status_label = QtLabel("")
        self.status_label.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        self.top_bar_layout.addWidget(self.status_label)

        # Add top bar at the very top
        self.layout.insertWidget(0, self.top_bar)

        # Image label inside a scroll area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        self.layout.addWidget(self.scroll_area)

        # Image path
        self.image_path = None
        # Threshold value
        self.rgb_threshold = 1
        # Radius multiplier for Avg. GL calculation
        self.gl_radius_multiplier = 0.5  # Default to 1/2 × r
        # STD requirement percentage
        self.std_requirement = 1.0  # Default 1%
        # Last folder path
        self.last_folder_path = os.path.expanduser("~")
    
    def load_settings(self):
        """Load saved settings from JSON file"""
        import json
        import os
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                
                # Restore threshold
                if 'rgb_threshold' in settings:
                    self.rgb_threshold = settings['rgb_threshold']
                    self.threshold_slider.setValue(self.rgb_threshold)
                    self.threshold_label.setText(f"RGB Threshold: {self.rgb_threshold}")
                
                # Restore radius multiplier
                if 'gl_radius_multiplier' in settings:
                    self.gl_radius_multiplier = settings['gl_radius_multiplier']
                    # Find and set the combo box index
                    for i in range(self.radius_combo.count()):
                        if self.radius_combo.itemData(i) == self.gl_radius_multiplier:
                            self.radius_combo.setCurrentIndex(i)
                            break
                
                # Restore STD requirement
                if 'std_requirement' in settings:
                    self.std_requirement = settings['std_requirement']
                    self.std_spinbox.setValue(self.std_requirement)
                
                # Restore last folder path
                if 'last_folder_path' in settings:
                    self.last_folder_path = settings['last_folder_path']
                
                # Restore window geometry
                if 'window_geometry' in settings:
                    geom = settings['window_geometry']
                    self.setGeometry(geom['x'], geom['y'], geom['width'], geom['height'])
                
                print(f"Settings loaded from {self.settings_file}")
        except Exception as e:
            print(f"Could not load settings: {e}")
    
    def save_settings(self):
        """Save current settings to JSON file"""
        import json
        try:
            settings = {
                'rgb_threshold': self.rgb_threshold,
                'gl_radius_multiplier': self.gl_radius_multiplier,
                'std_requirement': self.std_spinbox.value(),
                'last_folder_path': self.last_folder_path,
                'window_geometry': {
                    'x': self.geometry().x(),
                    'y': self.geometry().y(),
                    'width': self.geometry().width(),
                    'height': self.geometry().height()
                }
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Settings saved to {self.settings_file}")
        except Exception as e:
            print(f"Could not save settings: {e}")
    
    def process_folder_safe(self):
        """Wrapper to catch and display exceptions from process_folder"""
        try:
            self.process_folder()
        except Exception as e:
            import traceback
            from PyQt6.QtWidgets import QMessageBox
            error_msg = f"Error during folder processing:\n\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            self.progress_bar.setVisible(False)
    
    def display_image_in_gui(self, combined_img):
        """Display an image in the GUI image label.
        
        Args:
            combined_img: numpy array of image in RGB format
        """
        h, w, ch = combined_img.shape
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        max_w = int(screen_size.width() * 0.9)
        max_h = int(screen_size.height() * 0.9)
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            combined_disp = cv2.resize(combined_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = combined_disp.shape[:2]
        else:
            combined_disp = combined_img
        bytes_per_line = ch * w
        qimg = QImage(combined_disp.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()
        QApplication.processEvents()
    
    def process_folder(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QListWidget, QDialog, QVBoxLayout, QPushButton, QLabel
        import os
        import csv
        # Custom dialog for multi-folder selection
        from PyQt6.QtWidgets import QDialogButtonBox, QListWidgetItem
        # Step 1: Select main folder
        main_folder = QFileDialog.getExistingDirectory(self, "Select Main Folder", self.last_folder_path)
        if not main_folder:
            return
        
        # Save the selected folder path
        self.last_folder_path = main_folder
        self.save_settings()
        # Step 2: Check for images in main folder
        # Collect all variants: -g-00-org.png, -e-00-org.png, and -00-org.png
        # Process each variant separately (not as fallback)
        g_files = [f for f in os.listdir(main_folder) if f.lower().endswith('-g-00-org.png')]
        e_files = [f for f in os.listdir(main_folder) if f.lower().endswith('-e-00-org.png')]
        # For default, exclude files that are already g or e variants
        default_files = [f for f in os.listdir(main_folder) if f.lower().endswith('-00-org.png') and not f.lower().endswith('-g-00-org.png') and not f.lower().endswith('-e-00-org.png')]
        image_files = g_files + e_files + default_files
        subfolders = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
        folders = []
        if image_files:
            # Images in main folder, process just these
            folders = [main_folder]
        elif subfolders:
            # No images in main, but has subfolders: show dialog to select subfolders
            class SubfolderSelectDialog(QDialog):
                def __init__(self, subfolders, parent=None):
                    super().__init__(parent)
                    self.setWindowTitle("Select Subfolders to Process")
                    self.resize(500, 400)
                    self.selected = []
                    layout = QVBoxLayout(self)
                    self.label = QLabel("Select subfolders to process and click RUN:")
                    layout.addWidget(self.label)
                    self.list_widget = QListWidget()
                    self.list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
                    for folder in subfolders:
                        item = QListWidgetItem(os.path.basename(folder))
                        item.setData(256, folder)
                        self.list_widget.addItem(item)
                    layout.addWidget(self.list_widget)
                    self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                    layout.addWidget(self.button_box)
                    self.button_box.accepted.connect(self.accept)
                    self.button_box.rejected.connect(self.reject)
                def get_selected_folders(self):
                    return [self.list_widget.item(i).data(256) for i in range(self.list_widget.count()) if self.list_widget.item(i).isSelected()]
            dialog = SubfolderSelectDialog(subfolders, self)
            if not dialog.exec():
                return
            folders = dialog.get_selected_folders()
        else:
            QMessageBox.warning(self, "No Images or Subfolders", "No images or subfolders found in the selected folder.")
            return
        # Prepare for results
        import datetime
        all_results = []
        total_images = 0
        folder_image_lists = []
        for folder in folders:
            all_images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            # First try *-e-00-org.png pattern, then fall back to *-00-org.png
            image_files = [f for f in all_images if f.lower().endswith("-e-00-org.png")]
            if not image_files:
                image_files = [f for f in all_images if f.lower().endswith("-00-org.png")]
            print(f"Processing folder: {folder} ({len(image_files)} images out of {len(all_images)} images)")
            if len(image_files) > 0:
                print(f"  Sample files: {image_files[:3]}...")
            folder_image_lists.append((folder, image_files))
            total_images += len(image_files) * 2  # Count both original and calibrated (if exist)
        if total_images == 0:
            QMessageBox.warning(self, "No Images", "No image files found in the selected folders.")
            return
        self.progress_bar.setRange(0, total_images)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()
        processed = 0
        # Create CSV at the beginning with date_time prefix
        dt_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(os.getcwd(), f"{dt_prefix}_all_results.csv")
        print(f"CSV will be saved to: {csv_path}")
        fieldnames = ["folder", "filename", "r1", "r2", "r3", "r4", "d1", "d2", "sensor_usage", 
                     "Avg. GL Q1", "Avg. GL Q2", "Avg. GL Q3", "Avg. GL Q4",
                     "r1_cal", "r2_cal", "r3_cal", "r4_cal", "d1_cal", "d2_cal", "sensor_usage_cal",
                     "Avg. GL Q1 (Cal)", "Avg. GL Q2 (Cal)", "Avg. GL Q3 (Cal)", "Avg. GL Q4 (Cal)"]
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        import concurrent.futures
        def process_one_image(folder, fname, rgb_threshold, gl_radius_multiplier):
            import cv2, os
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            overlay, overlap_img, radii, overlap_internal_radius, ellipsoid_d1, ellipsoid_d2, percent_usage, avg_gray_levels = self.analyze_and_overlay(img, rgb_threshold, gl_radius_multiplier)
            return (folder, fname, overlay, overlap_img, radii, ellipsoid_d1, ellipsoid_d2, percent_usage, avg_gray_levels)

        # Prepare all jobs - both original and calibrated
        jobs = []
        for folder, image_files in folder_image_lists:
            output_dir = os.path.join(os.getcwd(), os.path.basename(folder) + "_outputs")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: {output_dir}")
            for fname in image_files:
                # Add original image job
                jobs.append((folder, fname, getattr(self, 'rgb_threshold', 16), getattr(self, 'gl_radius_multiplier', 0.5), output_dir, False))
                
                # Determine base filename and variant from original
                # Variants: -g-00-org.png, -e-00-org.png, or -00-org.png
                if fname.lower().endswith('-g-00-org.png'):
                    base_fname = fname.replace('-g-00-org.png', '', 1)
                    variant = 'g'
                elif fname.lower().endswith('-e-00-org.png'):
                    base_fname = fname.replace('-e-00-org.png', '', 1)
                    variant = 'e'
                else:
                    base_fname = fname.replace('-00-org.png', '', 1)
                    variant = 'default'
                
                # Check for calibrated version matching the same variant
                if variant == 'g':
                    cal_fname = f"{base_fname}-g-00-lsc-ccm.png"
                elif variant == 'e':
                    cal_fname = f"{base_fname}-e-00-lsc-ccm.png"
                else:
                    cal_fname = f"{base_fname}-00-lsc-ccm.png"
                
                cal_path = os.path.join(folder, cal_fname)
                if os.path.exists(cal_path):
                    jobs.append((folder, cal_fname, getattr(self, 'rgb_threshold', 16), getattr(self, 'gl_radius_multiplier', 0.5), output_dir, True))
        
        print(f"\nStarting processing of {len(jobs)} images (original + calibrated)...")

        results = [None] * len(jobs)
        processed = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(process_one_image, job[0], job[1], job[2], job[3]): (i, job) for i, job in enumerate(jobs)}
            for future in concurrent.futures.as_completed(future_to_idx):
                i, job = future_to_idx[future]
                try:
                    result = future.result()
                    results[i] = (job, result)
                except Exception as e:
                    print(f"  [ERROR] Failed to process {job[1]}: {e}")
                    import traceback
                    traceback.print_exc()
                    results[i] = (job, None)
                # Update progress bar in real-time
                processed += 1
                self.progress_bar.setValue(processed)
                self.status_label.setText(f"Processing: {job[1]} ({processed}/{len(jobs)})")
                QApplication.processEvents()
                print(f"  Processed {processed}/{len(jobs)}: {job[1]}")

        # Group results by original filename and variant
        processed = 0
        image_pairs = {}  # key: (base_filename, variant), value: {'org': result, 'cal': result}
        
        for (folder, fname, rgb_threshold, gl_radius_multiplier, output_dir, is_calibrated), result in results:
            if result is None:
                print(f"  [SKIP] Could not load image: {os.path.join(folder, fname)}")
                processed += 1
                self.progress_bar.setValue(processed)
                QApplication.processEvents()
                continue
            
            # Determine base filename and variant from either pattern
            if is_calibrated:
                # Handle calibrated: identify variant from suffix
                if fname.lower().endswith('-g-00-lsc-ccm.png'):
                    base_fname = fname.replace('-g-00-lsc-ccm.png', '', 1)
                    variant = 'g'
                elif fname.lower().endswith('-e-00-lsc-ccm.png'):
                    base_fname = fname.replace('-e-00-lsc-ccm.png', '', 1)
                    variant = 'e'
                else:
                    base_fname = fname.replace('-00-lsc-ccm.png', '', 1)
                    variant = 'default'
            else:
                # Handle original: identify variant from suffix
                if fname.lower().endswith('-g-00-org.png'):
                    base_fname = fname.replace('-g-00-org.png', '', 1)
                    variant = 'g'
                elif fname.lower().endswith('-e-00-org.png'):
                    base_fname = fname.replace('-e-00-org.png', '', 1)
                    variant = 'e'
                else:
                    base_fname = fname.replace('-00-org.png', '', 1)
                    variant = 'default'
            
            # Use (base_filename, variant) as key to separate g, e, and default variants
            key = (base_fname, variant)
            if key not in image_pairs:
                image_pairs[key] = {}
            
            image_pairs[key]['cal' if is_calibrated else 'org'] = {
                'folder': folder,
                'fname': fname,
                'output_dir': output_dir,
                'result': result
            }
        
        # Process each image pair (now including variants g, e, and default)
        for key, images in image_pairs.items():
            base_fname, variant = key  # Unpack (base_filename, variant) tuple
            for image_type, data in images.items():
                if data is None:
                    continue
                
                folder = data['folder']
                fname = data['fname']
                output_dir = data['output_dir']
                result = data['result']
                
                folder, fname, overlay, overlap_img, radii, ellipsoid_d1, ellipsoid_d2, percent_usage, avg_gray_levels = result
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                if overlap_img is not None:
                    left_img = overlay_rgb.copy()
                    right_img = overlap_img.copy()
                    lh = left_img.shape[0]
                    rh = right_img.shape[0]
                    target_h = min(lh, rh)
                    if lh != target_h:
                        scale = target_h / lh
                        lw = int(left_img.shape[1] * scale)
                        left_img = cv2.resize(left_img, (lw, target_h), interpolation=cv2.INTER_AREA)
                    if rh != target_h:
                        scale = target_h / rh
                        rw = int(right_img.shape[1] * scale)
                        right_img = cv2.resize(right_img, (rw, target_h), interpolation=cv2.INTER_AREA)
                    combined = np.hstack([left_img, right_img])
                else:
                    combined = overlay_rgb
                out_img_path = os.path.join(output_dir, os.path.splitext(fname)[0] + "_overlay.png")
                cv2.imwrite(out_img_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                print(f"  [SAVED] {out_img_path}")
                overlay_text = f"Sensor Usage: {percent_usage:.1f}%" if percent_usage is not None else "Sensor Usage: N/A"
                overlay_rgb_disp = overlay_rgb.copy()
                cv2.putText(overlay_rgb_disp, overlay_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 5, cv2.LINE_AA)

                # --- Display the combined overlay in the GUI ---
                self.display_image_in_gui(combined)

                # Store results for CSV writing (will be merged later)
                data['csv_ready'] = {
                    "r1": radii[0] if 0 < len(radii) else None,
                    "r2": radii[1] if 1 < len(radii) else None,
                    "r3": radii[2] if 2 < len(radii) else None,
                    "r4": radii[3] if 3 < len(radii) else None,
                    "d1": ellipsoid_d1,
                    "d2": ellipsoid_d2,
                    "sensor_usage": f"{percent_usage:.1f}" if percent_usage is not None else "",
                    "Avg. GL Q1": avg_gray_levels[0] if len(avg_gray_levels) > 0 else None,
                    "Avg. GL Q2": avg_gray_levels[1] if len(avg_gray_levels) > 1 else None,
                    "Avg. GL Q3": avg_gray_levels[2] if len(avg_gray_levels) > 2 else None,
                    "Avg. GL Q4": avg_gray_levels[3] if len(avg_gray_levels) > 3 else None,
                }
                
                processed += 1
                self.progress_bar.setValue(processed)
                QApplication.processEvents()
        
        # Write results to CSV
        for base_fname, images in image_pairs.items():
            result_row = {
                "folder": os.path.basename(images.get('org', images.get('cal'))['folder']),
                "filename": base_fname,
            }
            
            # Original image data
            if 'org' in images and 'csv_ready' in images['org']:
                org_data = images['org']['csv_ready']
                result_row.update({
                    "r1": org_data["r1"],
                    "r2": org_data["r2"],
                    "r3": org_data["r3"],
                    "r4": org_data["r4"],
                    "d1": org_data["d1"],
                    "d2": org_data["d2"],
                    "sensor_usage": org_data["sensor_usage"],
                    "Avg. GL Q1": org_data["Avg. GL Q1"],
                    "Avg. GL Q2": org_data["Avg. GL Q2"],
                    "Avg. GL Q3": org_data["Avg. GL Q3"],
                    "Avg. GL Q4": org_data["Avg. GL Q4"],
                })
            
            # Calibrated image data
            if 'cal' in images and 'csv_ready' in images['cal']:
                cal_data = images['cal']['csv_ready']
                result_row.update({
                    "r1_cal": cal_data["r1"],
                    "r2_cal": cal_data["r2"],
                    "r3_cal": cal_data["r3"],
                    "r4_cal": cal_data["r4"],
                    "d1_cal": cal_data["d1"],
                    "d2_cal": cal_data["d2"],
                    "sensor_usage_cal": cal_data["sensor_usage"],
                    "Avg. GL Q1 (Cal)": cal_data["Avg. GL Q1"],
                    "Avg. GL Q2 (Cal)": cal_data["Avg. GL Q2"],
                    "Avg. GL Q3 (Cal)": cal_data["Avg. GL Q3"],
                    "Avg. GL Q4 (Cal)": cal_data["Avg. GL Q4"],
                })
            
            all_results.append(result_row)
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result_row)
        
        # CSV is already written incrementally above
        # Show graph if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            df = pd.DataFrame(all_results)
            plt.figure(figsize=(10,6))
            for i in range(1,5):
                if f"r{i}" in df.columns:
                    plt.plot(df["filename"], df[f"r{i}"], marker='o', label=f"r{i} (Original)")
                if f"r{i}_cal" in df.columns and df[f"r{i}_cal"].notna().any():
                    plt.plot(df["filename"], df[f"r{i}_cal"], marker='s', linestyle='--', label=f"r{i} (Calibrated)")
            plt.xlabel("Image filename")
            plt.ylabel("Radius (pixels)")
            plt.title("Detected Radii per Quadrant (Original vs Calibrated)")
            plt.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            # Silently skip graph if matplotlib/pandas not available
            print(f"Could not show graph (matplotlib/pandas not installed): {e}")
        finally:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)

        # Hide progress bar when done
        self.progress_bar.setVisible(False)

    def on_threshold_changed(self, value):
        self.rgb_threshold = value
        self.threshold_label.setText(f"RGB Threshold: {value}")
        self.save_settings()  # Save when threshold changes
        if self.image_path:
            self.analyze_image()
    
    def on_radius_changed(self, index):
        """Called when radius multiplier dropdown changes"""
        self.gl_radius_multiplier = self.radius_combo.itemData(index)
        self.save_settings()  # Save when radius changes
        if self.image_path:
            self.open_image()
    
    def on_std_changed(self, value):
        """Called when STD requirement spinbox changes"""
        self.std_requirement = value
        self.save_settings()  # Save when STD changes
    
    def closeEvent(self, event):
        """Save settings when window is closed"""
        self.save_settings()
        event.accept()
    
    def analyze_image(self):
        from PyQt6.QtWidgets import QMessageBox
        import time
        if not self.image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        # Show indeterminate progress bar for single image analysis
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()
        try:
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return
            overlay, overlap_img, radii, overlap_internal_radius, ellipsoid_d1, ellipsoid_d2, percent_usage, avg_gray_levels = self.analyze_and_overlay(img, getattr(self, 'rgb_threshold', 8), getattr(self, 'gl_radius_multiplier', 0.5))
            # Apply CLAHE to the original image for better visibility (left image only)
            img_bgr = img.copy()
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            left_img = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
            # Overlay the left overlay on top of CLAHE image
            if overlay_rgb.shape[:2] != left_img.shape[:2]:
                overlay_rgb = cv2.resize(overlay_rgb, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_AREA)
            alpha_left = 0.2
            left_img = (overlay_rgb.astype(np.float32) * alpha_left + left_img.astype(np.float32) * (1 - alpha_left)).astype(np.uint8)
            if overlap_img is not None:
                right_img = overlap_img.copy()
                # Resize left and right to same height
                lh = left_img.shape[0]
                rh = right_img.shape[0]
                target_h = min(lh, rh)
                if lh != target_h:
                    scale = target_h / lh
                    lw = int(left_img.shape[1] * scale)
                    left_img = cv2.resize(left_img, (lw, target_h), interpolation=cv2.INTER_AREA)
                if rh != target_h:
                    scale = target_h / rh
                    rw = int(right_img.shape[1] * scale)
                    right_img = cv2.resize(right_img, (rw, target_h), interpolation=cv2.INTER_AREA)
                combined = np.hstack([left_img, right_img])
                h, w, ch = combined.shape
                screen = QApplication.primaryScreen()
                screen_size = screen.availableGeometry()
                max_w = int(screen_size.width() * 0.9)
                max_h = int(screen_size.height() * 0.9)
                scale = min(max_w / w, max_h / h, 1.0)
                if scale < 1.0:
                    combined = cv2.resize(combined, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    h, w = combined.shape[:2]
                bytes_per_line = ch * w
                qimg = QImage(combined.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pixmap)
                self.image_label.adjustSize()
            else:
                h, w, ch = left_img.shape
                bytes_per_line = ch * w
                qimg = QImage(left_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during analysis:\n{e}")
        finally:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)
    def analyze_csv_data(self):
        """Load CSV file and create normalized data visualization"""
        try:
            import os
            csv_file = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")[0]
            if not csv_file:
                return
            
            import csv
            import pandas as pd
            
            # Set matplotlib backend to Qt5Agg before importing pyplot
            import matplotlib
            matplotlib.use('Qt5Agg')
            
            # Load CSV file
            df = pd.read_csv(csv_file)
            
            # Validate required columns (original)
            required_cols = ["Avg. GL Q1", "Avg. GL Q2", "Avg. GL Q3", "Avg. GL Q4", "filename"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Missing Columns", f"CSV is missing required columns: {', '.join(missing_cols)}")
                return
            
            # Check if calibrated columns exist
            has_calibrated = all(col in df.columns for col in ["Avg. GL Q1 (Cal)", "Avg. GL Q2 (Cal)", "Avg. GL Q3 (Cal)", "Avg. GL Q4 (Cal)"])
            
            # Extract gray level data and validate
            q1_vals = df["Avg. GL Q1"].values
            q2_vals = df["Avg. GL Q2"].values
            q3_vals = df["Avg. GL Q3"].values
            q4_vals = df["Avg. GL Q4"].values
            filenames = df["filename"].values
            
            # Filter out rows with None/NaN values
            valid_mask = ~(pd.isna(q1_vals) | pd.isna(q2_vals) | pd.isna(q3_vals) | pd.isna(q4_vals) | (q1_vals == 0))
            if not valid_mask.any():
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Valid Data", "No valid gray level data found in CSV (all Q1 values are 0 or NaN)")
                return
            
            q1_vals = q1_vals[valid_mask]
            q2_vals = q2_vals[valid_mask]
            q3_vals = q3_vals[valid_mask]
            q4_vals = q4_vals[valid_mask]
            filenames = filenames[valid_mask]
            
            # Extract calibrated data if available
            q1_vals_cal = None
            q2_vals_cal = None
            q3_vals_cal = None
            q4_vals_cal = None
            if has_calibrated:
                q1_vals_cal = df["Avg. GL Q1 (Cal)"].values[valid_mask]
                q2_vals_cal = df["Avg. GL Q2 (Cal)"].values[valid_mask]
                q3_vals_cal = df["Avg. GL Q3 (Cal)"].values[valid_mask]
                q4_vals_cal = df["Avg. GL Q4 (Cal)"].values[valid_mask]
            
            # Calculate normalized values (Q2/Q1, Q3/Q1, Q4/Q1)
            norm_q2 = q2_vals / q1_vals
            norm_q3 = q3_vals / q1_vals
            norm_q4 = q4_vals / q1_vals
            
            # Calculate averages
            avg_norm_q2 = np.mean(norm_q2)
            avg_norm_q3 = np.mean(norm_q3)
            avg_norm_q4 = np.mean(norm_q4)
            
            # Calculate standard deviations
            std_norm_q2 = np.std(norm_q2)
            std_norm_q3 = np.std(norm_q3)
            std_norm_q4 = np.std(norm_q4)
            
            print(f"CSV Analysis Results (Original):")
            print(f"  Loaded {len(filenames)} valid images")
            print(f"  Average Q2/Q1: {avg_norm_q2:.3f} +/- {std_norm_q2:.3f}")
            print(f"  Average Q3/Q1: {avg_norm_q3:.3f} +/- {std_norm_q3:.3f}")
            print(f"  Average Q4/Q1: {avg_norm_q4:.3f} +/- {std_norm_q4:.3f}")
            
            # Calculate calibrated metrics if available
            norm_q2_cal = None
            norm_q3_cal = None
            norm_q4_cal = None
            avg_norm_q2_cal = None
            avg_norm_q3_cal = None
            avg_norm_q4_cal = None
            std_norm_q2_cal = None
            std_norm_q3_cal = None
            std_norm_q4_cal = None
            
            if has_calibrated and q1_vals_cal is not None:
                # Calculate normalized values for calibrated data
                norm_q2_cal = q2_vals_cal / q1_vals_cal
                norm_q3_cal = q3_vals_cal / q1_vals_cal
                norm_q4_cal = q4_vals_cal / q1_vals_cal
                
                # Calculate averages
                avg_norm_q2_cal = np.mean(norm_q2_cal)
                avg_norm_q3_cal = np.mean(norm_q3_cal)
                avg_norm_q4_cal = np.mean(norm_q4_cal)
                
                # Calculate standard deviations
                std_norm_q2_cal = np.std(norm_q2_cal)
                std_norm_q3_cal = np.std(norm_q3_cal)
                std_norm_q4_cal = np.std(norm_q4_cal)
                
                print(f"\nCSV Analysis Results (Calibrated):")
                print(f"  Average Q2/Q1 (Cal): {avg_norm_q2_cal:.3f} +/- {std_norm_q2_cal:.3f}")
                print(f"  Average Q3/Q1 (Cal): {avg_norm_q3_cal:.3f} +/- {std_norm_q3_cal:.3f}")
                print(f"  Average Q4/Q1 (Cal): {avg_norm_q4_cal:.3f} +/- {std_norm_q4_cal:.3f}")
            
            # Calculate Q3/Q2 and Q3/Q4 ratios
            norm_q3_q2 = q3_vals / q2_vals
            norm_q3_q4 = q3_vals / q4_vals
            avg_norm_q3_q2 = np.mean(norm_q3_q2)
            avg_norm_q3_q4 = np.mean(norm_q3_q4)
            std_norm_q3_q2 = np.std(norm_q3_q2)
            std_norm_q3_q4 = np.std(norm_q3_q4)
            
            print(f"  Average Q3/Q2: {avg_norm_q3_q2:.3f} +/- {std_norm_q3_q2:.3f}")
            print(f"  Average Q3/Q4: {avg_norm_q3_q4:.3f} +/- {std_norm_q3_q4:.3f}")
            
            # Calculate calibrated Q3/Q2 and Q3/Q4 if available
            norm_q3_q2_cal = None
            norm_q3_q4_cal = None
            avg_norm_q3_q2_cal = None
            avg_norm_q3_q4_cal = None
            std_norm_q3_q2_cal = None
            std_norm_q3_q4_cal = None
            
            if has_calibrated and q1_vals_cal is not None:
                norm_q3_q2_cal = q3_vals_cal / q2_vals_cal
                norm_q3_q4_cal = q3_vals_cal / q4_vals_cal
                avg_norm_q3_q2_cal = np.mean(norm_q3_q2_cal)
                avg_norm_q3_q4_cal = np.mean(norm_q3_q4_cal)
                std_norm_q3_q2_cal = np.std(norm_q3_q2_cal)
                std_norm_q3_q4_cal = np.std(norm_q3_q4_cal)
                
                print(f"  Average Q3/Q2 (Cal): {avg_norm_q3_q2_cal:.3f} +/- {std_norm_q3_q2_cal:.3f}")
                print(f"  Average Q3/Q4 (Cal): {avg_norm_q3_q4_cal:.3f} +/- {std_norm_q3_q4_cal:.3f}")
            
            # Create separate plots for each ratio
            self.create_single_ratio_plot(filenames, norm_q2, avg_norm_q2, std_norm_q2, 
                                         norm_q2_cal, avg_norm_q2_cal, std_norm_q2_cal, 
                                         "Q2/Q1", "blue")
            
            self.create_single_ratio_plot(filenames, norm_q3, avg_norm_q3, std_norm_q3, 
                                         norm_q3_cal, avg_norm_q3_cal, std_norm_q3_cal, 
                                         "Q3/Q1", "green")
            
            self.create_single_ratio_plot(filenames, norm_q4, avg_norm_q4, std_norm_q4, 
                                         norm_q4_cal, avg_norm_q4_cal, std_norm_q4_cal, 
                                         "Q4/Q1", "red")
            
            self.create_single_ratio_plot(filenames, norm_q3_q2, avg_norm_q3_q2, std_norm_q3_q2, 
                                         norm_q3_q2_cal, avg_norm_q3_q2_cal, std_norm_q3_q2_cal, 
                                         "Q3/Q2", "purple")
            
            self.create_single_ratio_plot(filenames, norm_q3_q4, avg_norm_q3_q4, std_norm_q3_q4, 
                                         norm_q3_q4_cal, avg_norm_q3_q4_cal, std_norm_q3_q4_cal, 
                                         "Q3/Q4", "orange")
            
            # Create GL Q1 analysis plot
            self.create_gl_q1_plot(filenames, q1_vals, q1_vals_cal if has_calibrated else None)
            
        except Exception as e:
            import traceback
            from PyQt6.QtWidgets import QMessageBox
            error_msg = f"Error during CSV analysis:\n\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
    
    def create_single_ratio_plot(self, filenames, norm_data, avg_val, std_val, 
                                 norm_data_cal, avg_val_cal, std_val_cal, 
                                 ratio_name, color):
        """Create a single plot for one normalized ratio with optional calibrated comparison"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Set matplotlib backend
            import matplotlib
            matplotlib.use('Qt5Agg')
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # X-axis: filename indices
            x = np.arange(len(filenames))
            
            # Get STD requirement percentage from GUI
            std_req = self.std_spinbox.value() if hasattr(self, 'std_spinbox') else 1.0
            std_multiplier = std_req / 100.0
            
            # Plot original data
            line_org = ax.plot(x, norm_data, color=color, linestyle='-', marker='o', label=f'{ratio_name} (Org)', linewidth=2, markersize=5)
            
            # Plot calibrated data if available
            if norm_data_cal is not None:
                line_cal = ax.plot(x, norm_data_cal, color=color, linestyle='--', marker='s', label=f'{ratio_name} (Cal)', linewidth=2, markersize=5, alpha=0.7)
            
            # Add horizontal average lines (Original - solid)
            ax.axhline(y=avg_val, color=color, linestyle='-', linewidth=2.5, alpha=0.8, label=f'Avg (Org): {avg_val:.3f}')
            
            # Add average line for calibrated if available
            if avg_val_cal is not None:
                ax.axhline(y=avg_val_cal, color=color, linestyle='--', linewidth=2.5, alpha=0.8, label=f'Avg (Cal): {avg_val_cal:.3f}')
            
            # Add STD tolerance bands for original
            if std_val is not None:
                upper = avg_val + (std_val * std_multiplier)
                lower = avg_val - (std_val * std_multiplier)
                ax.axhline(y=upper, color=color, linestyle=':', linewidth=2, alpha=0.6, label=f'+STD ({std_req}%)')
                ax.axhline(y=lower, color=color, linestyle=':', linewidth=2, alpha=0.6, label=f'-STD ({std_req}%)')
            
            # Add STD tolerance bands for calibrated if available
            if std_val_cal is not None and avg_val_cal is not None:
                upper_cal = avg_val_cal + (std_val_cal * std_multiplier)
                lower_cal = avg_val_cal - (std_val_cal * std_multiplier)
                ax.axhline(y=upper_cal, color=color, linestyle=':', linewidth=1.5, alpha=0.4)
                ax.axhline(y=lower_cal, color=color, linestyle=':', linewidth=1.5, alpha=0.4)
            
            # Add text annotations
            text_y = 0.98
            ax.text(0.02, text_y, f'{ratio_name} (Org): {avg_val:.3f} ± {std_val:.3f}', transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
            
            if avg_val_cal is not None:
                text_y -= 0.08
                ax.text(0.02, text_y, f'{ratio_name} (Cal): {avg_val_cal:.3f} ± {std_val_cal:.3f}', transform=ax.transAxes, 
                       fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
            
            # Labels and formatting
            ax.set_xlabel('Image Index', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{ratio_name} Ratio', fontsize=12, fontweight='bold')
            has_cal_text = " (Original & Calibrated)" if norm_data_cal is not None else ""
            title = f'{ratio_name} Normalized Ratio{has_cal_text} - STD Requirement: {std_req}%'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show every nth label to avoid crowding
            step = max(1, len(filenames) // 15)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([f"{i}" for i in range(0, len(filenames), step)], fontsize=9)
            
            # Add interactive tooltips with filenames
            try:
                import mplcursors
                # Create tooltip annotations for original data
                annotations_org = []
                for i, (xi, yi) in enumerate(zip(x, norm_data)):
                    annotations_org.append(f"Index {i}\n{filenames[i]}\nValue: {yi:.4f}")
                
                cursor_org = mplcursors.cursor(line_org[0], hover=True)
                cursor_org.connect("add", lambda sel: sel.annotation.set_text(
                    annotations_org[int(sel.index)] if int(sel.index) < len(annotations_org) else "N/A"))
                
                # Create tooltip annotations for calibrated data if available
                if norm_data_cal is not None:
                    annotations_cal = []
                    for i, (xi, yi) in enumerate(zip(x, norm_data_cal)):
                        annotations_cal.append(f"Index {i}\n{filenames[i]} (Cal)\nValue: {yi:.4f}")
                    
                    cursor_cal = mplcursors.cursor(line_cal[0], hover=True)
                    cursor_cal.connect("add", lambda sel: sel.annotation.set_text(
                        annotations_cal[int(sel.index)] if int(sel.index) < len(annotations_cal) else "N/A"))
            except ImportError:
                print("mplcursors not installed - tooltips unavailable. Install with: pip install mplcursors")
            
            plt.tight_layout()
            plt.show()
            
            print(f"{ratio_name} plot displayed successfully")
            
        except Exception as e:
            import traceback
            print(f"Error creating {ratio_name} plot: {e}")
            print(traceback.format_exc())
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Plot Error", f"Failed to create {ratio_name} plot:\n{e}")
    
    def create_normalized_plot(self, filenames, norm_q2, norm_q3, norm_q4, 
                              avg_q2, avg_q3, avg_q4,
                              std_q2=None, std_q3=None, std_q4=None,
                              norm_q2_cal=None, norm_q3_cal=None, norm_q4_cal=None,
                              avg_q2_cal=None, avg_q3_cal=None, avg_q4_cal=None,
                              std_q2_cal=None, std_q3_cal=None, std_q4_cal=None):
        """Create matplotlib plot with normalized quadrant ratios and STD bands"""
        try:
            import matplotlib.pyplot as plt
            # Don't force backend - use Qt5Agg which is compatible with PyQt6
            # matplotlib.use() should be called before importing pyplot
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # X-axis: filename indices
            x = np.arange(len(filenames))
            
            # Get STD requirement percentage from GUI
            std_req = self.std_spinbox.value() if hasattr(self, 'std_spinbox') else 1.0
            std_multiplier = std_req / 100.0  # Convert percentage to multiplier
            
            # Plot the three normalized ratio curves with markers (Original)
            line_q2 = ax.plot(x, norm_q2, 'b-o', label='Q2/Q1 (Org)', linewidth=2, markersize=4)
            line_q3 = ax.plot(x, norm_q3, 'g-s', label='Q3/Q1 (Org)', linewidth=2, markersize=4)
            line_q4 = ax.plot(x, norm_q4, 'r-^', label='Q4/Q1 (Org)', linewidth=2, markersize=4)
            
            # Plot calibrated curves if available
            if norm_q2_cal is not None:
                line_q2_cal = ax.plot(x, norm_q2_cal, 'b--d', label='Q2/Q1 (Cal)', linewidth=2, markersize=4, alpha=0.7)
                line_q3_cal = ax.plot(x, norm_q3_cal, 'g--x', label='Q3/Q1 (Cal)', linewidth=2, markersize=4, alpha=0.7)
                line_q4_cal = ax.plot(x, norm_q4_cal, 'r--+', label='Q4/Q1 (Cal)', linewidth=2, markersize=4, alpha=0.7)
            
            # Add horizontal average lines (Original - solid)
            ax.axhline(y=avg_q2, color='blue', linestyle='-', linewidth=2, alpha=0.7, label=f'Avg Q2/Q1 (Org): {avg_q2:.3f}')
            ax.axhline(y=avg_q3, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Avg Q3/Q1 (Org): {avg_q3:.3f}')
            ax.axhline(y=avg_q4, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Avg Q4/Q1 (Org): {avg_q4:.3f}')
            
            # Add average lines for calibrated data if available
            if avg_q2_cal is not None:
                ax.axhline(y=avg_q2_cal, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg Q2/Q1 (Cal): {avg_q2_cal:.3f}')
                ax.axhline(y=avg_q3_cal, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg Q3/Q1 (Cal): {avg_q3_cal:.3f}')
                ax.axhline(y=avg_q4_cal, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg Q4/Q1 (Cal): {avg_q4_cal:.3f}')
            
            # Add STD tolerance bands if standard deviations are provided
            if std_q2 is not None:
                upper_q2 = avg_q2 + (std_q2 * std_multiplier)
                lower_q2 = avg_q2 - (std_q2 * std_multiplier)
                ax.axhline(y=upper_q2, color='blue', linestyle=':', linewidth=2, alpha=0.5, label=f'Q2/Q1 +STD ({std_req}%)')
                ax.axhline(y=lower_q2, color='blue', linestyle=':', linewidth=2, alpha=0.5, label=f'Q2/Q1 -STD ({std_req}%)')
            
            if std_q3 is not None:
                upper_q3 = avg_q3 + (std_q3 * std_multiplier)
                lower_q3 = avg_q3 - (std_q3 * std_multiplier)
                ax.axhline(y=upper_q3, color='green', linestyle=':', linewidth=2, alpha=0.5, label=f'Q3/Q1 +STD ({std_req}%)')
                ax.axhline(y=lower_q3, color='green', linestyle=':', linewidth=2, alpha=0.5, label=f'Q3/Q1 -STD ({std_req}%)')
            
            if std_q4 is not None:
                upper_q4 = avg_q4 + (std_q4 * std_multiplier)
                lower_q4 = avg_q4 - (std_q4 * std_multiplier)
                ax.axhline(y=upper_q4, color='red', linestyle=':', linewidth=2, alpha=0.5, label=f'Q4/Q1 +STD ({std_req}%)')
                ax.axhline(y=lower_q4, color='red', linestyle=':', linewidth=2, alpha=0.5, label=f'Q4/Q1 -STD ({std_req}%)')
            
            # Add average values as text annotations
            ax.text(0.02, 0.98, f'Avg Q2/Q1 (Org): {avg_q2:.3f}', transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
            ax.text(0.02, 0.92, f'Avg Q3/Q1 (Org): {avg_q3:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
            ax.text(0.02, 0.86, f'Avg Q4/Q1 (Org): {avg_q4:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            
            if avg_q2_cal is not None:
                ax.text(0.02, 0.80, f'Avg Q2/Q1 (Cal): {avg_q2_cal:.3f}', transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.3))
                ax.text(0.02, 0.74, f'Avg Q3/Q1 (Cal): {avg_q3_cal:.3f}', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lime', alpha=0.3))
                ax.text(0.02, 0.68, f'Avg Q4/Q1 (Cal): {avg_q4_cal:.3f}', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            # Labels and formatting
            ax.set_xlabel('Image Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Normalized Gray Level Ratio', fontsize=12, fontweight='bold')
            has_cal_text = " (Original & Calibrated)" if avg_q2_cal is not None else ""
            title = f'Normalized Quadrant Ratios (Q/Q1){has_cal_text} - STD Requirement: {std_req}%'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8, ncol=3)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show every nth label to avoid crowding
            step = max(1, len(filenames) // 15)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([f"{i}" for i in range(0, len(filenames), step)], fontsize=9)
            
            # Add interactive tooltips with filenames
            try:
                import mplcursors
                # Create tooltip annotations for Q2/Q1
                annotations_q2 = []
                for i, (xi, yi) in enumerate(zip(x, norm_q2)):
                    annotations_q2.append(f"Index {i}\n{filenames[i]}\nQ2/Q1: {yi:.4f}")
                cursor_q2 = mplcursors.cursor(line_q2[0], hover=True)
                cursor_q2.connect("add", lambda sel: sel.annotation.set_text(
                    annotations_q2[int(sel.index)] if int(sel.index) < len(annotations_q2) else "N/A"))
                
                # Create tooltip annotations for Q3/Q1
                annotations_q3 = []
                for i, (xi, yi) in enumerate(zip(x, norm_q3)):
                    annotations_q3.append(f"Index {i}\n{filenames[i]}\nQ3/Q1: {yi:.4f}")
                cursor_q3 = mplcursors.cursor(line_q3[0], hover=True)
                cursor_q3.connect("add", lambda sel: sel.annotation.set_text(
                    annotations_q3[int(sel.index)] if int(sel.index) < len(annotations_q3) else "N/A"))
                
                # Create tooltip annotations for Q4/Q1
                annotations_q4 = []
                for i, (xi, yi) in enumerate(zip(x, norm_q4)):
                    annotations_q4.append(f"Index {i}\n{filenames[i]}\nQ4/Q1: {yi:.4f}")
                cursor_q4 = mplcursors.cursor(line_q4[0], hover=True)
                cursor_q4.connect("add", lambda sel: sel.annotation.set_text(
                    annotations_q4[int(sel.index)] if int(sel.index) < len(annotations_q4) else "N/A"))
                
                # Add tooltips for calibrated curves if available
                if norm_q2_cal is not None:
                    annotations_q2_cal = []
                    for i, (xi, yi) in enumerate(zip(x, norm_q2_cal)):
                        annotations_q2_cal.append(f"Index {i}\n{filenames[i]} (Cal)\nQ2/Q1: {yi:.4f}")
                    cursor_q2_cal = mplcursors.cursor(line_q2_cal[0], hover=True)
                    cursor_q2_cal.connect("add", lambda sel: sel.annotation.set_text(
                        annotations_q2_cal[int(sel.index)] if int(sel.index) < len(annotations_q2_cal) else "N/A"))
                    
                    annotations_q3_cal = []
                    for i, (xi, yi) in enumerate(zip(x, norm_q3_cal)):
                        annotations_q3_cal.append(f"Index {i}\n{filenames[i]} (Cal)\nQ3/Q1: {yi:.4f}")
                    cursor_q3_cal = mplcursors.cursor(line_q3_cal[0], hover=True)
                    cursor_q3_cal.connect("add", lambda sel: sel.annotation.set_text(
                        annotations_q3_cal[int(sel.index)] if int(sel.index) < len(annotations_q3_cal) else "N/A"))
                    
                    annotations_q4_cal = []
                    for i, (xi, yi) in enumerate(zip(x, norm_q4_cal)):
                        annotations_q4_cal.append(f"Index {i}\n{filenames[i]} (Cal)\nQ4/Q1: {yi:.4f}")
                    cursor_q4_cal = mplcursors.cursor(line_q4_cal[0], hover=True)
                    cursor_q4_cal.connect("add", lambda sel: sel.annotation.set_text(
                        annotations_q4_cal[int(sel.index)] if int(sel.index) < len(annotations_q4_cal) else "N/A"))
            except ImportError:
                print("mplcursors not installed - tooltips unavailable. Install with: pip install mplcursors")
            
            plt.tight_layout()
            plt.show()
            
            print("Plot displayed successfully")
            
        except Exception as e:
            import traceback
            print(f"Error creating plot: {e}")
            print(traceback.format_exc())
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Plot Error", f"Failed to create plot:\n{e}")

    def create_gl_q1_plot(self, filenames, q1_vals, q1_vals_cal=None):
        """Create plot with Avg GL Q1 vs device and histogram (with calibrated comparison if available)"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure with 2 subplots (1 row, 2 columns)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # X-axis: image indices
            x = np.arange(len(filenames))
            
            # Left plot: Avg GL Q1 vs Image Index
            line_q1 = ax1.plot(x, q1_vals, 'b-o', linewidth=2, markersize=5, label='Avg GL Q1 (Org)')
            avg_q1 = np.mean(q1_vals)
            ax1.axhline(y=avg_q1, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg (Org): {avg_q1:.1f}')
            
            # Plot calibrated data if available
            if q1_vals_cal is not None:
                line_q1_cal = ax1.plot(x, q1_vals_cal, 'r-s', linewidth=2, markersize=5, label='Avg GL Q1 (Cal)')
                avg_q1_cal = np.mean(q1_vals_cal)
                ax1.axhline(y=avg_q1_cal, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg (Cal): {avg_q1_cal:.1f}')
            
            ax1.set_xlabel('Image Index', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Avg GL Q1 (Gray Level)', fontsize=12, fontweight='bold')
            title_suffix = " (Original vs Calibrated)" if q1_vals_cal is not None else ""
            ax1.set_title(f'Average Gray Level Q1 vs Device{title_suffix}', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Set x-axis to show every nth label to avoid crowding
            step = max(1, len(filenames) // 15)
            ax1.set_xticks(x[::step])
            ax1.set_xticklabels([f"{i}" for i in range(0, len(filenames), step)], fontsize=9)
            
            # Add interactive tooltips with filenames for left plot
            try:
                import mplcursors
                # Create tooltip annotations for original
                annotations_q1 = []
                for i, (xi, yi) in enumerate(zip(x, q1_vals)):
                    annotations_q1.append(f"Index {i}\n{filenames[i]}\nAvg GL Q1: {yi:.1f}")
                cursor_q1 = mplcursors.cursor(line_q1[0], hover=True)
                cursor_q1.connect("add", lambda sel: sel.annotation.set_text(
                    annotations_q1[int(sel.index)] if int(sel.index) < len(annotations_q1) else "N/A"))
                
                # Create tooltip annotations for calibrated
                if q1_vals_cal is not None:
                    annotations_q1_cal = []
                    for i, (xi, yi) in enumerate(zip(x, q1_vals_cal)):
                        annotations_q1_cal.append(f"Index {i}\n{filenames[i]} (Cal)\nAvg GL Q1: {yi:.1f}")
                    cursor_q1_cal = mplcursors.cursor(line_q1_cal[0], hover=True)
                    cursor_q1_cal.connect("add", lambda sel: sel.annotation.set_text(
                        annotations_q1_cal[int(sel.index)] if int(sel.index) < len(annotations_q1_cal) else "N/A"))
            except ImportError:
                print("mplcursors not installed - tooltips unavailable. Install with: pip install mplcursors")
            
            # Right plot: Histogram of Avg GL Q1
            ax2.hist(q1_vals, bins=20, color='blue', edgecolor='black', alpha=0.6, label='Original')
            avg_q1 = np.mean(q1_vals)
            ax2.axvline(x=avg_q1, color='blue', linestyle='--', linewidth=2.5, label=f'Mean (Org): {avg_q1:.1f}')
            std_q1 = np.std(q1_vals)
            ax2.axvline(x=avg_q1 + std_q1, color='blue', linestyle=':', linewidth=2, alpha=0.7)
            ax2.axvline(x=avg_q1 - std_q1, color='blue', linestyle=':', linewidth=2, alpha=0.7)
            
            # Plot calibrated histogram if available
            if q1_vals_cal is not None:
                ax2.hist(q1_vals_cal, bins=20, color='red', edgecolor='black', alpha=0.6, label='Calibrated')
                avg_q1_cal = np.mean(q1_vals_cal)
                ax2.axvline(x=avg_q1_cal, color='red', linestyle='--', linewidth=2.5, label=f'Mean (Cal): {avg_q1_cal:.1f}')
                std_q1_cal = np.std(q1_vals_cal)
                ax2.axvline(x=avg_q1_cal + std_q1_cal, color='red', linestyle=':', linewidth=2, alpha=0.7)
                ax2.axvline(x=avg_q1_cal - std_q1_cal, color='red', linestyle=':', linewidth=2, alpha=0.7)
            
            ax2.set_xlabel('Avg GL Q1 Value', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax2.set_title(f'Histogram of Avg GL Q1{title_suffix}', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Print statistics
            print(f"\nAvg GL Q1 Statistics (Original):")
            print(f"  Mean: {avg_q1:.2f}")
            print(f"  Std Dev: {std_q1:.2f}")
            print(f"  Min: {np.min(q1_vals):.2f}")
            print(f"  Max: {np.max(q1_vals):.2f}")
            print(f"  Median: {np.median(q1_vals):.2f}")
            
            if q1_vals_cal is not None:
                print(f"\nAvg GL Q1 Statistics (Calibrated):")
                print(f"  Mean: {avg_q1_cal:.2f}")
                print(f"  Std Dev: {std_q1_cal:.2f}")
                print(f"  Min: {np.min(q1_vals_cal):.2f}")
                print(f"  Max: {np.max(q1_vals_cal):.2f}")
                print(f"  Median: {np.median(q1_vals_cal):.2f}")
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            import traceback
            print(f"Error creating GL Q1 plot: {e}")
            print(traceback.format_exc())
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Plot Error", f"Failed to create GL Q1 plot:\n{e}")

    def draw_transparent_circle(self, image, center, radius, color, fill=False):
        # Always fill the circle with 50% opacity for the left-side image
        overlay = image.copy()
        output = image.copy()
        b, g, r, a = color
        # Fill with 50% opacity regardless of 'fill' argument
        fill_alpha = 0.5
        cv2.circle(overlay, center, radius, (b, g, r), -1)
        cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)
        # Draw outline as before (with original alpha)
        if not fill:
            overlay2 = output.copy()
            cv2.circle(overlay2, center, radius, (b, g, r), 2)
            cv2.addWeighted(overlay2, a / 255.0, output, 1 - a / 255.0, 0, output)
        return output

        # ...existing code...

    def open_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
            self.analyze_image()  # Automatically analyze when image is loaded
        else:
            self.image_path = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
