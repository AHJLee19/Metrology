import os
import cv2
import matplotlib.pyplot as plt

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img

def interactive_point_selection(img, num_points=2, title="Select points"):
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
