import cv2
import numpy as np

def compute_homography(img_src, img_dst, checkerboard_size):
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    ret_src, corners_src = cv2.findChessboardCornersSB(gray_src, checkerboard_size)
    ret_dst, corners_dst = cv2.findChessboardCornersSB(gray_dst, checkerboard_size)
    if not (ret_src and ret_dst):
        return None
    H, _ = cv2.findHomography(corners_src, corners_dst, cv2.RANSAC)
    return H

def warp_points(points, H):
    pts = np.array(points, dtype=np.float32).reshape(-1,1,2)
    warped_pts = cv2.perspectiveTransform(pts, H)
    return warped_pts.reshape(-1,2).tolist()

def warp_image(img, H, shape):
    return cv2.warpPerspective(img, H, shape)

def overlay_images(img1, img2):
    return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

def stitch_images(img_left, img_right, H_left_to_right):
    h1,w1 = img_left.shape[:2]
    h2,w2 = img_right.shape[:2]
    corners_left = np.array([[0,0],[0,h1],[w1,h1],[w1,0]],dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_left, H_left_to_right)
    all_corners = np.vstack((warped_corners, np.array([[0,0],[0,h2],[w2,h2],[w2,0]],dtype=np.float32).reshape(-1,1,2)))
    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())
    translation = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
    panorama = cv2.warpPerspective(img_left, translation @ H_left_to_right, (xmax-xmin, ymax-ymin))
    panorama[-ymin:h2-ymin, -xmin:w2-xmin] = img_right
    return panorama
