import cv2  
import matplotlib.pyplot as plt
import numpy as np

image_path = './pikachu.png'
gray_image = cv2.imread(image_path,0)
image = cv2.resize(gray_image,(512,512))

msb_image = image & 224 #E0 or 11100000

difference_image = cv2.absdiff(image,msb_image)

plt.figure(figsize=(7,7))
plt.subplot(2,2,1)
plt.title('Original image')
plt.imshow(image,cmap='gray')

plt.subplot(2,2,2)
plt.title('MSB-3 image')
plt.imshow(msb_image,cmap='gray')

plt.subplot(2,2,3)
plt.title('Difference image')
plt.imshow(difference_image,cmap='gray')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

