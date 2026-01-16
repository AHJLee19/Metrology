import cv2
import numpy as np

def stitch_images(img1, img2, feature_detector='ORB', ratio_thresh=0.75):
    """
    Stitch two images together using feature matching and homography.

    Parameters:
    - img1, img2: input images (BGR)
    - feature_detector: 'ORB', 'SIFT', or 'SURF'
    - ratio_thresh: Lowe's ratio for filtering matches

    Returns:
    - stitched_img: combined image
    """
    # Step 1: Initialize feature detector
    if feature_detector == 'ORB':
        detector = cv2.ORB_create(5000)
    elif feature_detector == 'SIFT':
        detector = cv2.SIFT_create()
    elif feature_detector == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    else:
        raise ValueError("Invalid feature detector")

    # Step 2: Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Step 3: Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if feature_detector=='ORB' else cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Step 4: Apply Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Not enough matches to compute homography")

    # Step 5: Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Step 6: Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Step 7: Warp img1 to img2's plane
    h2, w2 = img2.shape[:2]
    h1, w1 = img1.shape[:2]

    # Calculate output size
    corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)
    all_corners = np.concatenate((warped_corners_img1, np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    H_translation = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    stitched_img = cv2.warpPerspective(img1, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    stitched_img[translation[1]:h2+translation[1], translation[0]:w2+translation[0]] = img2

    return stitched_img


if __name__ == "__main__":
    # Example usage
    img1 = cv2.imread("../data/unknown_parts/img1.jpg")
    img2 = cv2.imread("../data/unknown_parts/img2.jpg")
    stitched = stitch_images(img1, img2)
    cv2.imwrite("stitched_result.jpg", stitched)
    print("Stitched image saved as 'stitched_result.jpg'")
