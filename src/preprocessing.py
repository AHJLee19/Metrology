import cv2
import numpy as np

def detect_checkerboard_corners(img, pattern_size=(8, 8)):
    """
    Detect checkerboard corners in an image.

    Parameters:
    - img: input image (BGR or grayscale)
    - pattern_size: tuple (rows, cols) of internal checkerboard corners

    Returns:
    - corners: detected corners in order (numpy array of shape (num_corners, 1, 2))
    - ret: True if corners found, False otherwise
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        # Refine corner positions
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    return corners, ret


def align_off_plane(img, src_points=None, pattern_size=(8,8), square_size_cm=10, output_size=None):
    """
    Align image to remove off-plane distortion using checkerboard detection.

    Parameters:
    - img: input image (BGR)
    - src_points: manually specified 4 points (optional)
    - pattern_size: checkerboard internal corners (rows, cols)
    - square_size_cm: size of one square in cm (used to define dst rectangle)
    - output_size: (width, height) of output image (optional)

    Returns:
    - aligned_img: perspective-corrected image
    """
    # Step 1: Detect corners if not provided
    if src_points is None:
        corners, found = detect_checkerboard_corners(img, pattern_size)
        if not found:
            raise ValueError("Checkerboard not detected and no src_points provided")
        # Pick the 4 outer corners: top-left, top-right, bottom-right, bottom-left
        rows, cols = pattern_size
        tl = corners[0,0]
        tr = corners[cols-1,0]
        br = corners[-1,0]
        bl = corners[-cols,0]
        src_points = np.array([tl, tr, br, bl], dtype=np.float32)

    else:
        src_points = np.array(src_points, dtype=np.float32)

    # Step 2: Define destination rectangle
    if output_size is None:
        width = square_size_cm * pattern_size[1]  # cols
        height = square_size_cm * pattern_size[0]  # rows
        output_size = (int(width), int(height))
    dst_points = np.array([[0,0],
                           [output_size[0]-1, 0],
                           [output_size[0]-1, output_size[1]-1],
                           [0, output_size[1]-1]], dtype=np.float32)

    # Step 3: Compute homography and warp image
    H, _ = cv2.findHomography(src_points, dst_points)
    aligned_img = cv2.warpPerspective(img, H, output_size)

    return aligned_img


if __name__ == "__main__":
    # Example usage
    img_path = "../data/calibration_board/board1.jpg"
    img = cv2.imread(img_path)

    try:
        aligned = align_off_plane(img, pattern_size=(8,8), square_size_cm=10)
        cv2.imwrite("aligned_board.jpg", aligned)
        print("Aligned image saved as 'aligned_board.jpg'")
    except ValueError as e:
        print(e)
