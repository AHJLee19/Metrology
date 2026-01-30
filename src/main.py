import cv2
import matplotlib.pyplot as plt

from preprocessing import load_image, interactive_point_selection
from calibration import find_checkerboard_scale
from metrics import compute_distance_mm
from image_registration import compute_homography, warp_image

# ---------------- CONFIG ----------------
CHECKERBOARD = (3, 6)     # inner corners
SQUARE_SIZE_CM = 5

left_image_path = "data/calibration_board/Metrology_Left.jpeg"
right_image_path = "data/calibration_board/Metrology_Right.jpeg"

# ---------------- LOAD ----------------
img_left = load_image(left_image_path)
img_right = load_image(right_image_path)

# ---------------- CALIBRATION ----------------
scale_left = find_checkerboard_scale(img_left, CHECKERBOARD, SQUARE_SIZE_CM)
scale_right = find_checkerboard_scale(img_right, CHECKERBOARD, SQUARE_SIZE_CM)

# ---------------- MEASURE IN EACH IMAGE ----------------
print("Select measurement points on LEFT image")
pts_left = interactive_point_selection(img_left, title="LEFT measurement")
dist_left = compute_distance_mm(pts_left[0], pts_left[1], scale_left)
print(f"Distance on LEFT image: {dist_left:.2f} mm\n")

print("Select measurement points on RIGHT image")
pts_right = interactive_point_selection(img_right, title="RIGHT measurement")
dist_right = compute_distance_mm(pts_right[0], pts_right[1], scale_right)
print(f"Distance on RIGHT image: {dist_right:.2f} mm\n")

# ---------------- HOMOGRAPHY ----------------
H_L2R = compute_homography(img_left, img_right, CHECKERBOARD)
H_R2L = compute_homography(img_right, img_left, CHECKERBOARD)

if H_L2R is not None and H_R2L is not None:
    # Warp left -> right
    img_left_in_right = cv2.warpPerspective(img_left, H_L2R, (img_right.shape[1], img_right.shape[0]))
    # Warp right -> left
    img_right_in_left = cv2.warpPerspective(img_right, H_R2L, (img_left.shape[1], img_left.shape[0]))

    # Plot both transformations
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.title("Original LEFT Image")
    plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 2)
    plt.title("Original RIGHT Image")
    plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 3)
    plt.title("LEFT Image Warped to RIGHT")
    plt.imshow(cv2.cvtColor(img_left_in_right, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 4)
    plt.title("RIGHT Image Warped to LEFT")
    plt.imshow(cv2.cvtColor(img_right_in_left, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()
else:
    print("Homography could not be computed for one or both images.")
