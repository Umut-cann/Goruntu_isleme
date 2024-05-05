import numpy as np
import matplotlib.pyplot as plt
import cv2

def standard_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def s_curve_contrast(image):

    normalized_image = image / 255.0


    s_curve_image = standard_sigmoid(normalized_image)


    s_curve_image = (s_curve_image - np.min(s_curve_image)) / (np.max(s_curve_image) - np.min(s_curve_image))


    s_curve_image = (s_curve_image * 255.0).astype(np.uint8)

    return s_curve_image

image = cv2.imread('input.png',)


plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orijinal Görüntü')
plt.axis('off')


contrast_enhanced_image = s_curve_contrast(image)
plt.subplot(1, 2, 2)
plt.imshow(contrast_enhanced_image, cmap='gray')
plt.title('Kontrast Güçlendirilmiş Görüntü')
plt.axis('off')

plt.show()
