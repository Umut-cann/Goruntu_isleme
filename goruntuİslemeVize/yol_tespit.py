import cv2
import numpy as np


image = cv2.imread('yol.jpg')


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])


yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)


edges = cv2.Canny(yellow_mask, 50, 150)


lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=5)


if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


cv2.imshow('Detected Yellow Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
