import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# Initialize webcam and modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\gupta\Downloads\fuckit\Sign-Language-Interpreter\Model\model.h5", r"C:\Users\gupta\Downloads\fuckit\Sign-Language-Interpreter\Model\labels.txt")

# Parameters
offset = 20
imgSize = 224  # Change the image size to 224

folder = r"C:\Users\gupta\Downloads\fuckit\nothing"
counter = 0

# Label definitions
labels = ["A","B","C",]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:  # Portrait
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize to (wCal, imgSize)
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        else:  # Landscape or square
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize to (imgSize, hCal)
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        # Display prediction
        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    # Show the output image
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

# Release resources on exit
cap.release()
cv2.destroyAllWindows()
