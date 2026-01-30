import cv2
import numpy as np

def find_checkerboard_scale(img, checkerboard_size, square_size_cm):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(gray, checkerboard_size)
    if ret:
        corners = corners.squeeze()
        pt1, pt2 = corners[0], corners[1]
        pixel_dist = np.linalg.norm(pt1 - pt2)
        scale_mm_per_pixel = (square_size_cm * 10) / pixel_dist
        return scale_mm_per_pixel
    else:
        raise RuntimeError("Checkerboard corners not found.")
