import cv2  # necessary libraries for cv2 for OpenCV functions &
import numpy as np  #for numerical operations
import matplotlib.pyplot as plt

image_path = './pikachu.png'
gray_image = cv2.imread(image_path,0)
image = cv2.resize(gray_image,(512,512))

gamma = 2 #gamma value for the powe law transformation.The gamma value controls the amount of gamma correction applied to the image. Adjust this value to achieve different results in terms of brightness and contrast.

c = 1   #255/(np.log(1+np.max(image)))   #define parameter c for inverse logarithmic transformation

power_law_trasnformed = np.power(image/255.0,gamma)*255.0   #It raises each pixel value in the image to the power of gamma, scales the result to the range [0, 255]
power_law_trasnformed = np.uint8(power_law_trasnformed)
#pixel values are always non negative so we use unsigned 8-bit integer(uint8)

inverse_log_transformed = c*np.log(1+image/255.0)*255.0
inverse_log_transformed = np.uint8(inverse_log_transformed)

diff_image = cv2.absdiff(power_law_trasnformed,inverse_log_transformed)


plt.figure(figsize=(7,7))
plt.subplot(2,2,1)
plt.title('Original Image')
plt.imshow(image,cmap='gray')

plt.subplot(2,2,2)
plt.title('Power Law Transform')
plt.imshow(power_law_trasnformed,cmap='gray')

plt.subplot(2,2,3)
plt.title('Inverse Logarithmic Transform')
plt.imshow(inverse_log_transformed,cmap='gray')

plt.subplot(2,2,4)
plt.title('Difference Image')
plt.imshow(diff_image,cmap='gray')

plt.show()

cv2.waitKey(1000)
cv2.destroyAllWindows()