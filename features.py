import cv2
import numpy as np

IMAGE_SIZE = 128

def extract_features_from_image(image):
    """Extract simple color, shape and texture features from an image array."""
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Color information
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_features = []

    for channel in cv2.split(hsv):
        color_features.extend([
            float(np.mean(channel)),
            float(np.std(channel))
        ])

    # Edge density
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # Brightness/contrast
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))

    # Small grayscale representation
    small = cv2.resize(gray, (16, 16)).astype("float32") / 255.0

    # Gradient orientation captures object edges and shape without exact pixels.
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    shape_features = []
    for row in range(0, IMAGE_SIZE, 16):
        for column in range(0, IMAGE_SIZE, 16):
            cell_magnitude = magnitude[row:row + 16, column:column + 16]
            cell_angle = angle[row:row + 16, column:column + 16] % 180
            histogram, _ = np.histogram(
                cell_angle,
                bins=9,
                range=(0, 180),
                weights=cell_magnitude
            )
            total = np.sum(histogram)
            if total > 0:
                histogram = histogram / total
            shape_features.extend(histogram.tolist())

    # Histograms preserve material/color patterns without depending on exact pixels.
    hist_features = []
    for channel in cv2.split(hsv):
        histogram = cv2.calcHist([channel], [0], None, [16], [0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        hist_features.extend(histogram.tolist())

    features = np.concatenate([
        np.array(color_features, dtype=np.float32),
        np.array([edge_density, gray_mean, gray_std], dtype=np.float32),
        small.flatten(),
        np.array(shape_features, dtype=np.float32),
        np.array(hist_features, dtype=np.float32)
    ])

    return features

def extract_features(image_path):
    """Extract simple color, shape and texture features using OpenCV."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("OpenCV could not read the image.")

    return extract_features_from_image(image)
