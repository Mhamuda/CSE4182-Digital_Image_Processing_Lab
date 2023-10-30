import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = './Aerial Image.tif'
gray_image = cv2.imread(image_path,0)
image = cv2.resize(gray_image,(512,512))

power_law = image.copy()
inverse_log = image.copy()
gamma = 0.5     #gamma value for the powe law transformation.The gamma value controls the amount of gamma correction applied to the image. Adjust this value to achieve different results in terms of brightness and contrast.

power_law = np.power(power_law/255.0,gamma)*255.0       #It raises each pixel value in the image to the power of gamma, scales the result to the range [0, 255]
power_law = np.uint8(power_law)     #pixel values are always non negative so we use unsigned 8-bit integer(uint8)

c = 255.0/np.log(1+255.0)
inverse_log = np.exp(inverse_log/c)-1
inverse_log = np.uint8(inverse_log)

diff_image = cv2.absdiff(power_law,inverse_log)
diff_image = np.uint8(diff_image)

plt.figure(figsize=(7,7))
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)

plt.subplot(2,2,1)
plt.title('Original image')
plt.imshow(image,cmap='gray')

plt.subplot(2,2,2)
plt.title('Powe law image')
plt.imshow(power_law,cmap='gray')

plt.subplot(2,2,3)
plt.title('Inverse logarithmic image')
plt.imshow(inverse_log,cmap='gray')

plt.subplot(2,2,4)
plt.title('Difference image')
plt.imshow(diff_image,cmap='gray')

plt.show()
