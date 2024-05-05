import cv2
import numpy as np


image = cv2.imread('yuz3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=40, minRadius=30, maxRadius=40)


if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image, center, radius, (0, 255, 0), 2)


cv2.imshow('Detected Eyes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


