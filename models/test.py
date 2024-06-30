import numpy as np
import cv2

# Combine line segments (assuming lines is a list of line segments)
def combine_line_segments(lines):
    # Combine line segments to form a single continuous line
    combined_line = []
    for line in lines:
        combined_line.extend(line)
    return np.array(combined_line)

# Interpolate points along the lines
def interpolate_points(line, num_points=100):
    # Interpolate points along the line
    x_values = np.linspace(line[0][0], line[-1][0], num_points)
    y_values = np.linspace(line[0][1], line[-1][1], num_points)
    points = np.column_stack((x_values, y_values))
    return points

# Apply Canny edge detection
def edge_detection(image):
    # Convert image to grayscale
   # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Example usage
# Assuming 'lines' contains the line segments of the geometric shape
lines = [[(100, 100), (200, 100)], [(200, 100), (200, 200)], [(200, 200), (100, 200)], [(100, 200), (100, 100)]]
combined_line = combine_line_segments(lines)
interpolated_points = interpolate_points(combined_line)
# Create a black canvas
canvas = np.zeros((300, 300), dtype=np.uint8)
# Draw the shape on the canvas
cv2.polylines(canvas, [np.int32(interpolated_points)], isClosed=True, color=255, thickness=1)
# Apply edge detection
edges = edge_detection(canvas)
# Display the results

cv2.imshow('Original Shape', canvas)
cv2.imshow('Edges Detected', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()