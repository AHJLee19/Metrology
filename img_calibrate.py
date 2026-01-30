import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# CONFIGURATION
# Checkerboard settings for your 4x7 squares (inner corners = squares - 1)
CHECKERBOARD = (3, 6)  # width x height in inner corners
SQUARE_SIZE_CM = 5     # one square in cm

# Image paths
left_image_path = "data/calibration_board/Metrology_Left.jpeg"
right_image_path = "data/calibration_board/Metrology_Right.jpeg"


# UTILITY FUNCTIONS
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img

def interactive_point_selection(img, num_points=2, title="Select points"):
    """Manually select points on the image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    plt.title(title)
    
    selected_points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None and len(selected_points) < num_points:
            selected_points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=5)
            fig.canvas.draw()
            print(f"Selected point: ({event.xdata:.2f}, {event.ydata:.2f})")
        if len(selected_points) == num_points:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return selected_points

def find_checkerboard_scale(img, checkerboard_size, square_size_cm):
    """
    Attempts to detect the checkerboard and compute mm/pixel scale.
    If detection fails, falls back to manual selection of two adjacent corners.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improve contrast

    # Try automatic detection first
    ret, corners = cv2.findChessboardCornersSB(gray, checkerboard_size)
    if ret:
        corners = corners.squeeze()
        print(f"Checkerboard detected automatically. Using first two corners for scale.")
        pt1, pt2 = corners[0], corners[1]
    else:
        print("Automatic checkerboard detection failed. Please select two adjacent corners manually.")
        selected_points = interactive_point_selection(img, num_points=2, title="Select two adjacent corners for scale")
        pt1, pt2 = np.array(selected_points[0]), np.array(selected_points[1])
    
    pixel_dist = np.linalg.norm(pt1 - pt2)
    scale_mm_per_pixel = (square_size_cm * 10) / pixel_dist  # convert cm -> mm
    print(f"Pixel distance: {pixel_dist:.2f}, Scale: {scale_mm_per_pixel:.4f} mm/pixel")
    return scale_mm_per_pixel

def compute_distance_mm(pt1, pt2, scale_mm_per_pixel):
    """Compute Euclidean distance between two points in mm."""
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pixel_dist = np.linalg.norm(pt1 - pt2)
    return pixel_dist * scale_mm_per_pixel


# MAIN WORKFLOW
# Load images
img_left = load_image(left_image_path)
img_right = load_image(right_image_path)

# Compute scale for both images
scale_left = find_checkerboard_scale(img_left, CHECKERBOARD, SQUARE_SIZE_CM)
scale_right = find_checkerboard_scale(img_right, CHECKERBOARD, SQUARE_SIZE_CM)

# Select points on LEFT image and compute distance
print("\nSelect points on LEFT image for measurement")
points_left = interactive_point_selection(img_left, num_points=2, title="LEFT image: select two points")
dist_left_mm = compute_distance_mm(points_left[0], points_left[1], scale_left)
print(f"Distance on LEFT image: {dist_left_mm:.2f} mm")

# Select points on RIGHT image and compute distance
print("\nSelect points on RIGHT image for measurement")
points_right = interactive_point_selection(img_right, num_points=2, title="RIGHT image: select two points")
dist_right_mm = compute_distance_mm(points_right[0], points_right[1], scale_right)
print(f"Distance on RIGHT image: {dist_right_mm:.2f} mm")

# Optional: compute homography to map LEFT -> RIGHT
def compute_homography(img_src, img_dst, checkerboard_size=CHECKERBOARD):
    """Compute homography using checkerboard corners (if visible)."""
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    ret_src, corners_src = cv2.findChessboardCornersSB(gray_src, checkerboard_size)
    ret_dst, corners_dst = cv2.findChessboardCornersSB(gray_dst, checkerboard_size)
    if not (ret_src and ret_dst):
        print("Checkerboard not found in both images. Homography cannot be computed automatically.")
        return None
    corners_src = corners_src.squeeze()
    corners_dst = corners_dst.squeeze()
    H, _ = cv2.findHomography(corners_src, corners_dst, cv2.RANSAC)
    return H

# Compute homographies
H_left_to_right = compute_homography(img_left, img_right, CHECKERBOARD)
H_right_to_left = compute_homography(img_right, img_left, CHECKERBOARD)

if H_left_to_right is not None and H_right_to_left is not None:
    # Warp left -> right
    img_left_in_right = cv2.warpPerspective(img_left, H_left_to_right, (img_right.shape[1], img_right.shape[0]))
    # Warp right -> left
    img_right_in_left = cv2.warpPerspective(img_right, H_right_to_left, (img_left.shape[1], img_left.shape[0]))

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
