import matplotlib.pyplot as plt
from skimage import io
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet
from skimage.restoration import estimate_sigma

image_path = "Skeleton.jpg"
noisy_img = img_as_float(io.imread(image_path, as_gray=True))
ref_img = img_as_float(io.imread(image_path, as_gray=True))

# Denoising using bilateral filter
denoise_bilateral_img = denoise_bilateral(noisy_img, sigma_spatial=15)
bilateral_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_bilateral_img)

# Denoising using total variation (TV) filter
denoise_tv_img = denoise_tv_chambolle(noisy_img, weight=0.1)
TV_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_tv_img)

# Denoising using wavelet filter
denoise_wavelet_img = denoise_wavelet(noisy_img)
Wavelet_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_wavelet_img)

print("PSNR of input noisy image = inf")
print("PSNR of bilateral cleaned image = ", bilateral_cleaned_psnr)
print("PSNR of TV cleaned image = ", TV_cleaned_psnr)
print("PSNR of wavelet cleaned image = ", Wavelet_cleaned_psnr)

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Bilateral Cleaned Image')
plt.imshow(denoise_bilateral_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('TV Cleaned Image')
plt.imshow(denoise_tv_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Wavelet Cleaned Image')
plt.imshow(denoise_wavelet_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
