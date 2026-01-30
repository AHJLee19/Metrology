import cv2
import numpy as np
from preprocessing import interactive_point_selection

def find_checkerboard_scale(img, checkerboard_size, square_size_cm):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    ret, corners = cv2.findChessboardCornersSB(gray, checkerboard_size)

    if ret:
        corners = corners.squeeze()
        print("Checkerboard detected automatically.")
        pt1, pt2 = corners[0], corners[1]
    else:
        print("Checkerboard detection failed. Select two adjacent corners manually.")
        pts = interactive_point_selection(
            img, num_points=2, title="Select adjacent checkerboard corners"
        )
        pt1, pt2 = np.array(pts[0]), np.array(pts[1])

    pixel_dist = np.linalg.norm(pt1 - pt2)
    scale_mm_per_pixel = (square_size_cm * 10) / pixel_dist

    print(f"Scale: {scale_mm_per_pixel:.4f} mm/pixel\n")
    return scale_mm_per_pixel
