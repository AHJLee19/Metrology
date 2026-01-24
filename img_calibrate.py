import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load calibration image
road_image_dir = '/Users/gtsechpe/Documents/GTSECHPE/road/data_gopro/mp4/092224calibration_far'
image_files = sorted(os.listdir(road_image_dir))
index = 0
image_file = image_files[index]
img_path = os.path.join(road_image_dir, image_file)
img = cv2.imread(img_path)

# Use checkerboard of 6 (width) x 10 (height) squares of 10cm x 10cm
# Define the size of the checkerboard (number of inner corners per row and column)
width_of_checkerboard = 5
height_of_checkerboard = 9
checkerboard_size = (width_of_checkerboard, height_of_checkerboard)
# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the checkerboard corners
ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
if ret:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    # Extract the corners and plot them
    corners = corners.squeeze()  # Remove extra dimensions
    plt.plot(corners[:, 0], corners[:, 1], 'r.', markersize=1)  # Plot red circles at each corner
    plt.title("Checkerboard Corners")
    plt.show()
else:
    print("Checkerboard corners not found")

square_size_cm = 10  #centimeters
# Calculate the pixel distance between two adjacent corners
pixel_dist = np.linalg.norm(corners[0] - corners[1])
# Calculate the scale factor (physical distance per pixel)
scale_factor = square_size_cm / pixel_dist
point1 = np.array([x1, y1])
point2 = np.array([x2, y2])
# Calculate the pixel distance between the two points
pixel_distance = np.linalg.norm(point1 - point2)
# Convert the pixel distance to physical distance
physical_distance = pixel_distance * scale_factor
print(f"Physical distance between the points: {physical_distance} cm")




# INTERACTIVE (2) POINT SELECTION 
# requires: pip3 install ipympl
%matplotlib inline
%matplotlib widget
# Load the image again
img = cv2.imread(img_path)
# Convert the image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Create the figure and axes
fig, ax = plt.subplots()
ax.imshow(img_rgb)
plt.title("Click to Select Two Points")
# List to store the selected points
selected_points = []
# Mouse click event handler
def onclick(event):
    if event.xdata is not None and event.ydata is not None and len(selected_points) < 2:
        selected_points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ro')  # Mark the point
        fig.canvas.draw()  # Update the figure
        print(f"Selected point: ({event.xdata}, {event.ydata})")
    
    if len(selected_points) == 2:
        print("Two points selected. Disconnecting event.")
        fig.canvas.mpl_disconnect(cid)

# Connect the event
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

#Check point selection    
fig, ax = plt.subplots()
ax.imshow(img_rgb)
for point in selected_points:
    ax.plot(point[0], point[1], 'ro', markersize=5) 
plt.show() 

x1 = selected_points[0][0]
y1 = selected_points[0][1]
x2 = selected_points[1][0]
y2 = selected_points[1][1]
point1 = np.array([x1, y1])
point2 = np.array([x2, y2])







import os
import numpy as np
import re
from PIL import Image, ImageFilter
import cv2
from sklearn.cluster import DBSCAN
import random
import math
from scipy.signal import find_peaks
from collections import Counter
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 0: calibration

calibration_image_path = '/Users/gavriil/Documents/GAVRIIL/PROJECTS_PURDUE/Cyclist/pycode/gopro_calibration'
def distance_calibration(calibration_image_path):
    image_files = sorted(os.listdir(calibration_image_path))
    index = 0
    image_file = image_files[index]
    img_path = os.path.join(calibration_image_path, image_file)
    img = cv2.imread(img_path)
    # Use checkerboard of 6 (width) x 10 (height) squares of 10cm x 10cm
    # Define the size of the checkerboard (number of inner corners per row and column)
    width_of_checkerboard = 5
    height_of_checkerboard = 9
    checkerboard_size = (width_of_checkerboard, height_of_checkerboard)
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if ret:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.imshow(img_rgb)
        # Extract the corners and plot them
        corners = corners.squeeze()  # Remove extra dimensions
        #plt.plot(corners[:, 0], corners[:, 1], 'r.', markersize=1)  # Plot red circles at each corner
        #plt.title("Checkerboard Corners")
        #plt.show()
    else:
        print("Checkerboard corners not found")

    square_size_cm = 10  #centimeters
    # Calculate the pixel distance between two adjacent corners
    pixel_dist = np.linalg.norm(corners[0] - corners[1])
    # Calculate the scale factor (physical distance per pixel)
    scale_factor = square_size_cm / pixel_dist
    #point1 = np.array([x1, y1])
    #point2 = np.array([x2, y2])
    # Calculate the pixel distance between the two points
    #pixel_distance = np.linalg.norm(point1 - point2)
    # Convert the pixel distance to physical distance
    #physical_distance = pixel_distance * scale_factor
    #print(f"Physical distance between the points: {physical_distance} cm")
    return scale_factor
scale_factor = distance_calibration(calibration_image_path)