from preprocessing import load_image
from calibration import find_checkerboard_scale
from image_registration import compute_homography, warp_points, warp_image, overlay_images, stitch_images
from metrics import compute_distance_mm
from main import interactive_point_selection
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
CHECKERBOARD = (3, 6)  # 4x7 squares → inner corners = 3x6
SQUARE_SIZE_CM = 10

img_left = load_image("data\calibration_board\Metrology_Left.jpeg")
img_right = load_image("data\calibration_board\Metrology_Right.jpeg")

# ---------------------------
# CALIBRATION
# ---------------------------
scale_left = find_checkerboard_scale(img_left, CHECKERBOARD, SQUARE_SIZE_CM)
scale_right = find_checkerboard_scale(img_right, CHECKERBOARD, SQUARE_SIZE_CM)

# ---------------------------
# HOMOGRAPHY
# ---------------------------
H_left_to_right = compute_homography(img_left, img_right, CHECKERBOARD)
if H_left_to_right is None:
    raise RuntimeError("Homography could not be computed.")

# ---------------------------
# MANUAL POINT SELECTION ON LEFT
# ---------------------------
points_left = interactive_point_selection(img_left, num_points=2)

# Map points to RIGHT image
points_right = warp_points(points_left, H_left_to_right)

# ---------------------------
# DISTANCE CALCULATION
# ---------------------------
dist_left_mm = compute_distance_mm(points_left[0], points_left[1], scale_left)
dist_right_mm = compute_distance_mm(points_right[0], points_right[1], scale_right)
print(f"Distance LEFT: {dist_left_mm:.2f} mm")
print(f"Distance RIGHT: {dist_right_mm:.2f} mm")

# ---------------------------
# IMAGE TRANSFORMATION AND OVERLAY
# ---------------------------
img_left_in_right = warp_image(img_left, H_left_to_right, (img_right.shape[1], img_right.shape[0]))
overlay_in_right = overlay_images(img_left_in_right, img_right)

# ---------------------------
# IMAGE STITCHING
# ---------------------------
panorama = stitch_images(img_left, img_right, H_left_to_right)

# ---------------------------
# VISUALIZATION
# ---------------------------
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.title("LEFT Image")
plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))

plt.subplot(2,2,2)
plt.title("RIGHT Image")
plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))

plt.subplot(2,2,3)
plt.title("Overlay LEFT → RIGHT")
plt.imshow(cv2.cvtColor(overlay_in_right, cv2.COLOR_BGR2RGB))

plt.subplot(2,2,4)
plt.title("Stitched Panorama")
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
