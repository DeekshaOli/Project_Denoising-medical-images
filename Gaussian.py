import numpy as np
import cv2
from matplotlib import pyplot as plt


def denoise_image(image, sigma):
    filtered_image = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma, sigmaY=sigma)
    return filtered_image

if __name__ == "__main__":
    image = cv2.imread("Skeleton.jpg")
    noisy_image = image + np.random.normal(0, 0.1, image.shape)
    denoised_image = denoise_image(noisy_image, sigma=5)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Denoised Image using Gaussian.')
    plt.imshow(denoised_image[:, :, 0], cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
