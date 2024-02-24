import matplotlib.pyplot as plt
from transform import *

# lung image output to test out
lung_image = Image.open('../Data/00000001_000.png')
histogram_eq = HistogramEqual()
hist_eq = histogram_eq(lung_image)
# Display the original and the equalized image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(lung_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Histogram Equalized Image')
plt.imshow(hist_eq, cmap='gray')
plt.axis('off')

plt.show()