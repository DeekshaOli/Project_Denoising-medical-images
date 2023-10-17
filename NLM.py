import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Skeleton.jpg')

denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img[:, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image using NLM.')
plt.imshow(denoised[:, :, 0], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
