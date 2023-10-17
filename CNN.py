import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load the noisy image
image_path = 'Input.jpg'
noisy_image = load_img(image_path, color_mode='grayscale')
noisy_image = img_to_array(noisy_image)
noisy_image = noisy_image.astype('float32') / 255.0

# Preprocess the image
noisy_image = np.expand_dims(noisy_image, axis=0)

# Define the denoising CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(228, 228, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(noisy_image, noisy_image, epochs=50, batch_size=1)

# Denoise the image
denoised_image = model.predict(noisy_image)

# Rescale the denoised image
denoised_image = denoised_image[0] * 255.0
denoised_image = np.clip(denoised_image, 0, 255).astype('uint8')

# Calculate pixel-wise accuracy
original_image = load_img(image_path, color_mode='grayscale')
original_image = img_to_array(original_image)
original_image = original_image.astype('float32') / 255.0

accuracy = np.mean(np.abs(original_image - denoised_image))

print("Pixel-wise Accuracy: {:.2%}".format(1 - accuracy))

# Save the denoised image
output_path = 'Denoised_Skeleton.png'
tf.keras.preprocessing.image.save_img(output_path, denoised_image, scale=True)

# Display the original and denoised images
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_image[:, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image using CNN')
plt.imshow(denoised_image[:, :, 0], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
