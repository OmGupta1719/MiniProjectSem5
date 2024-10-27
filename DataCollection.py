import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow detection of two hands

# Parameters
offset = 20
imgSize = 224  # Change the image size to 224
folder = r"C:\Users\gupta\Downloads\fuckit\nothing\C"  # Use raw string for path
counter = 0

# Ensure the folder exists
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        for hand in hands:  # Loop through all detected hands
            x, y, w, h = hand['bbox']

            # Ensure the crop does not go out of bounds
            x_start = max(x - offset, 0)
            y_start = max(y - offset, 0)
            x_end = min(x + w + offset, img.shape[1])
            y_end = min(y + h + offset, img.shape[0])

            imgCrop = img[y_start:y_end, x_start:x_end]  # Crop the image safely

            # Create a white canvas and resize the cropped image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White canvas
            imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))  # Resize to 224x224

            # Place the resized image on the white canvas
            imgWhite = imgCrop

            # Display the cropped and processed images for each hand
            cv2.imshow(f"ImageCrop Hand {hands.index(hand) + 1}", imgCrop)
            cv2.imshow(f"ImageWhite Hand {hands.index(hand) + 1}", imgWhite)

    # Show the original image
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        # Save the white image
        cv2.imwrite(f'{folder}/Image_{counter}_{time.time()}.jpg', imgWhite)
        print(f'Image {counter} saved!')

    # Condition to stop the loop (press 'q' to quit)
    if key == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close windows on exit
cap.release()
cv2.destroyAllWindows()
