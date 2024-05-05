import cv2
import numpy as np
import matplotlib.pyplot as plt

def deblurring_algorithm(blurred_image_path):

    image = cv2.imread(blurred_image_path)


    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    flow = cv2.calcOpticalFlowFarneback(prev=gray_image, next=gray_image, flow=None, pyr_scale=0.5, levels=5,
                                         winsize=15, iterations=20, poly_n=7, poly_sigma=1.5,
                                         flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)


    horizontal_flow = flow[..., 0]
    vertical_flow = flow[..., 1]


    kernel_radius = np.sqrt(horizontal_flow ** 2 + vertical_flow ** 2).mean()


    kernel_radius = int(kernel_radius)


    kernel = np.zeros((2 * kernel_radius + 1, 2 * kernel_radius + 1), np.float32)


    sigma = 20
    for i in range(-kernel_radius, kernel_radius + 1):
        for j in range(-kernel_radius, kernel_radius + 1):
            x = i + kernel_radius
            y = j + kernel_radius
            distance = np.sqrt(x ** 2 + y ** 2)
            weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
            kernel[i, j] = weight


    kernel /= np.sum(kernel)


    deblurred_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    return deblurred_image

blurred_image_path = 'blurlu.png'
blurred_image = cv2.imread(blurred_image_path)
deblurred_image = deblurring_algorithm(blurred_image_path)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Blurred Image')
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Deblurred Image')
plt.imshow(cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
