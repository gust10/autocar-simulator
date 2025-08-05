import cv2
import numpy as np
import math

def preprocess(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White color mask
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 25, 255))
    return mask


def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    # Only keep bottom half or a trapezoid ahead of the car
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.6), int(height * 0.2)),
        (int(width * 0.4), int(height * 0.2)),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_edges(masked_image):
    edges = cv2.Canny(masked_image, 50, 150)
    return edges

def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=50, maxLineGap=150)
    return lines

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

def overlay(original, line_img):
    return cv2.addWeighted(original, 0.8, line_img, 1, 1)

def lane_finding_pipeline(frame):
    mask = preprocess(frame)
    edges = detect_edges(mask)
    lines = detect_lines(edges)
    line_img = display_lines(frame, lines)
    final = overlay(frame, line_img)
    return edges

def classify_8_directions(edges):
    h, w = edges.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # Define color map for 8 directions
    direction_colors = [
        (255, 0, 255),    # dir0: top-left
        (255, 0, 0),      # dir1: top
        (255, 255, 0),    # dir2: top-right
        (0, 255, 0),      # dir3: left
        (0, 255, 255),    # dir4: right
        (0, 128, 255),    # dir5: bottom-left
        (0, 0, 255),      # dir6: bottom
        (255, 255, 255),  # dir7: bottom-right
    ]

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]

    for y in range(1, h-1):
        for x in range(1, w-1):
            if edges[y, x] == 255:
                for idx, (dy, dx) in enumerate(directions):
                    ny, nx = y + dy, x + dx
                    if edges[ny, nx] == 255:
                        output[y, x] = direction_colors[idx]
                        break  # Assign only first matched direction

    return output


def calculate_steering_angle(image, max_angle_deg=30):
    """
    image: camera frame (numpy array)
    max_angle_deg: maximum angle in degrees the steering can turn left/right.
    Returns: steering angle in degrees (negative = left, positive = right)
    """

    if image is None:
        print("No image captured from camera.")
        return 0

    # Convert to HSV and threshold white line
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 25, 255))  # white threshold

    # Visualize
    output = image.copy()
    height, width = image.shape[:2]
    center_x = width // 2

    frame = output.copy()

     # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150) # 100, 200
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # kernel size = (5,5)

    classified_edges = classify_8_directions(edges)

    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(blurred,
                                      maxCorners=100,
                                      qualityLevel=0.9,
                                      minDistance=10)
    if corners is not None:
        corners = corners.astype(np.intp)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Show the result
    cv2.imshow('Corners in Video', frame)

    # Moments to find center of white area
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        error = cx - center_x

        # Normalize error to [-1, 1]
        normalized_error = error / center_x

        # Convert to degrees
        steering_angle_deg = -normalized_error * max_angle_deg

        # Draw visuals
        cv2.circle(output, (cx, height - 30), 5, (0, 255, 0), -1)  # Green dot
        cv2.line(output, (center_x, 0), (center_x, height), (255, 0, 0), 1)  # Center line
        cv2.line(output, (center_x, height - 30), (cx, height - 30), (0, 0, 255), 2)  # Error line

        # Show angle
        cv2.putText(output, f"Steering Angle: {steering_angle_deg:.2f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show images
    cv2.imshow("Original with Steering Info", output)
    cv2.imshow("White Mask", cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)) #mask

    resized_image1 = cv2.resize(classified_edges, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('8-Neighbor Classified', resized_image1)
    cv2.waitKey(1)

    output2 = lane_finding_pipeline(image)

    resized_image = cv2.resize(output2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Lane Detection", resized_image)

    return steering_angle_deg