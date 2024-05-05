import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy

image = cv2.imread('tarla.jpg')


lower_green = np.array([25, 50, 50])
upper_green = np.array([100, 255, 255])
mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower_green, upper_green)


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


data = {
    'No': [],
    'Center': [],
    'Length': [],
    'Width': [],
    'Diagonal': [],
    'Energy': [],
    'Entropy': [],
    'Mean': [],
    'Median': []
}


for i, contour in enumerate(contours):

    x, y, w, h = cv2.boundingRect(contour)


    center = (x + w // 2, y + h // 2)


    data['No'].append(i + 1)
    data['Center'].append(center)
    data['Length'].append(w)
    data['Width'].append(h)
    data['Diagonal'].append(np.sqrt(w ** 2 + h ** 2))
    data['Energy'].append(np.sum(image[y:y + h, x:x + w] ** 2))
    hist = cv2.calcHist([image[y:y + h, x:x + w]], [0], None, [256], [0, 256])
    data['Entropy'].append(entropy(hist, base=2))
    data['Mean'].append(np.mean(image[y:y + h, x:x + w]))
    data['Median'].append(np.median(image[y:y + h, x:x + w]))


df = pd.DataFrame(data)


df.to_excel('hiperspektral_ozellikler.xlsx', index=False)

