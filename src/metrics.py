import numpy as np

def compute_distance_mm(pt1, pt2, scale_mm_per_pixel):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pixel_dist = np.linalg.norm(pt1 - pt2)
    return pixel_dist * scale_mm_per_pixel
