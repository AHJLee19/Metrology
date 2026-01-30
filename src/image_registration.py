import cv2
import numpy as np

def compute_homography(img_src, img_dst, checkerboard_size):
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    ret_src, corners_src = cv2.findChessboardCornersSB(gray_src, checkerboard_size)
    ret_dst, corners_dst = cv2.findChessboardCornersSB(gray_dst, checkerboard_size)

    if not (ret_src and ret_dst):
        print("Checkerboard not found in both images.")
        return None

    H, _ = cv2.findHomography(corners_src, corners_dst, cv2.RANSAC)
    return H

def warp_image(img, H, shape):
    return cv2.warpPerspective(img, H, shape)
