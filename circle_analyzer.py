# IMPORTANT: All code changes must be thoroughly checked, run, and validated for correctness and robustness before presenting to the user. Never provide untested or incomplete solutions.

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QPalette, QColor, QImage, QPainter, QPen, QBrush
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def analyze_and_overlay(self, img, rgb_threshold):
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
        radii = [r for (_, _, r, _) in centers_radii]
        return overlay, overlap_img, radii, overlap_internal_radius, d1, d2, percent_usage
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Circle Analyzer")
        self.setGeometry(100, 100, 1000, 800)
        self.set_dark_theme()
        self.init_ui()

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
        from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QScrollArea, QSlider, QLabel as QtLabel, QProgressBar
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
        self.folder_button.clicked.connect(self.process_folder)
        self.top_bar_layout.addWidget(self.folder_button)

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

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate by default
        self.progress_bar.setVisible(False)
        self.top_bar_layout.addWidget(self.progress_bar)

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
    def process_folder(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QListWidget, QDialog, QVBoxLayout, QPushButton, QLabel
        import os
        import csv
        import matplotlib.pyplot as plt
        # Custom dialog for multi-folder selection
        from PyQt6.QtWidgets import QDialogButtonBox, QListWidgetItem
        # Step 1: Select main folder
        main_folder = QFileDialog.getExistingDirectory(self, "Select Main Folder", os.path.expanduser("~"))
        if not main_folder:
            return
        # Step 2: Check for images in main folder
        image_files = [f for f in os.listdir(main_folder) if f.lower().endswith('-00-org.png')  and f.lower().startswith('c')]
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
            image_files = [f for f in all_images if f.lower().endswith("00-org.png") and f.lower().startswith('c')]
            print(f"Processing folder: {folder} ({len(image_files)} images out of {len(all_images)} images)")
            folder_image_lists.append((folder, image_files))
            total_images += len(image_files)
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
        fieldnames = ["folder", "filename", "r1", "r2", "r3", "r4", "d1", "d2", "sensor_usage"]
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        import concurrent.futures
        def process_one_image(folder, fname, rgb_threshold):
            import cv2, os
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            overlay, overlap_img, radii, overlap_internal_radius, ellipsoid_d1, ellipsoid_d2, percent_usage = self.analyze_and_overlay(img, rgb_threshold)
            return (folder, fname, overlay, overlap_img, radii, ellipsoid_d1, ellipsoid_d2, percent_usage)

        # Prepare all jobs
        jobs = []
        for folder, image_files in folder_image_lists:
            output_dir = os.path.join(os.getcwd(), os.path.basename(folder) + "_outputs")
            os.makedirs(output_dir, exist_ok=True)
            for fname in image_files:
                jobs.append((folder, fname, getattr(self, 'rgb_threshold', 16), output_dir))

        results = [None] * len(jobs)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(process_one_image, job[0], job[1], job[2]): (i, job) for i, job in enumerate(jobs)}
            for future in concurrent.futures.as_completed(future_to_idx):
                i, job = future_to_idx[future]
                result = future.result()
                results[i] = (job, result)

        processed = 0
        for (folder, fname, rgb_threshold, output_dir), result in results:
            if result is None:
                print(f"  [SKIP] Could not load image: {os.path.join(folder, fname)}")
                processed += 1
                self.progress_bar.setValue(processed)
                QApplication.processEvents()
                continue
            folder, fname, overlay, overlap_img, radii, ellipsoid_d1, ellipsoid_d2, percent_usage = result
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
            overlay_text = f"Sensor Usage: {percent_usage:.1f}%" if percent_usage is not None else "Sensor Usage: N/A"
            overlay_rgb_disp = overlay_rgb.copy()
            cv2.putText(overlay_rgb_disp, overlay_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 5, cv2.LINE_AA)

            # --- Display the combined overlay in the GUI ---
            h, w, ch = combined.shape
            screen = QApplication.primaryScreen()
            screen_size = screen.availableGeometry()
            max_w = int(screen_size.width() * 0.9)
            max_h = int(screen_size.height() * 0.9)
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                combined_disp = cv2.resize(combined, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                h, w = combined_disp.shape[:2]
            else:
                combined_disp = combined
            bytes_per_line = ch * w
            qimg = QImage(combined_disp.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()
            QApplication.processEvents()

            result_row = {
                "folder": os.path.basename(folder),
                "filename": fname,
                **{f"r{i+1}": radii[i] if i < len(radii) else None for i in range(4)},
                "d1": ellipsoid_d1,
                "d2": ellipsoid_d2,
                "sensor_usage": f"{percent_usage:.1f}" if percent_usage is not None else ""
            }
            all_results.append(result_row)
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result_row)
            processed += 1
            self.progress_bar.setValue(processed)
            QApplication.processEvents()
        # CSV is already written incrementally above
        # Show graph if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            df = pd.DataFrame(all_results)
            plt.figure(figsize=(10,6))
            for i in range(1,5):
                plt.plot(df["filename"], df[f"r{i}"], marker='o', label=f"r{i}")
            plt.plot(df["filename"], df["internal_r"], marker='x', label="internal_r", linewidth=3, color='black')
            plt.xlabel("Image filename")
            plt.ylabel("Radius (pixels)")
            plt.title("Detected Radii and Internal Overlap Radius")
            plt.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.information(self, "Graph Error", f"Could not show graph: {e}")
        finally:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)

        # Hide progress bar when done
        self.progress_bar.setVisible(False)

    def on_threshold_changed(self, value):
        self.rgb_threshold = value
        self.threshold_label.setText(f"RGB Threshold: {value}")
        if self.image_path:
            self.analyze_image()
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
            overlay, overlap_img, radii, overlap_internal_radius, ellipsoid_d1, ellipsoid_d2, percent_usage = self.analyze_and_overlay(img, getattr(self, 'rgb_threshold', 8))
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
