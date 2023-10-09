# led_detection_py
import cv2
import numpy as np

# Load your image
image = cv2.imread("your_image.jpg")

# Convert the image to grayscale for better processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to create a binary image
_, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize lists to store centroid coordinates and areas
centroid_list = []
area_list = []

# Loop over the detected contours
for i, contour in enumerate(contours):
    # Calculate the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        centroid = (centroid_x, centroid_y)
    else:
        centroid = (0, 0)

    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Append centroid coordinates and area to the respective lists
    centroid_list.append(centroid)
    area_list.append(area)

    # Draw the bright spot on the original image
    cv2.circle(image, centroid, 5, (0, 0, 255), -1)  # Red circle

# Save the output image with bright spots marked
cv2.imwrite("led_detection_results.png", image)

# Open a text file for writing
with open("led_detection_results.txt", "w") as file:
    # Write the number of LEDs detected to the file
    num_leds = len(centroid_list)
    file.write(f"No. of LEDs detected: {num_leds}\n")

    # Loop over the detected LEDs and write centroid coordinates and area to the file
    for i, (centroid, area) in enumerate(zip(centroid_list, area_list)):
        file.write(f"LED #{i + 1} - Centroid: {centroid}, Area: {area}\n")

# File is automatically closed when the "with" block exits
