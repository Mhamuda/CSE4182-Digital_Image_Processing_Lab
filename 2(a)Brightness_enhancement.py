import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = './pikachu.png'
gray_image = cv2.imread(image_path,0)
image = cv2.resize(gray_image,(512,512))

enhanced_image = image.copy()

min_graylevel = 10
max_graylevel = 100

height,width = image.shape

for i in range(0,height,1):
    for j in range(0,width,1):
        pixel_value = enhanced_image[i][j]

        if min_graylevel<=pixel_value<=max_graylevel:
            enhanced_image[i][j] =min(255,pixel_value+50)

plt.figure(figsize=(7,7))
plt.subplot(2,1,1)
plt.title('Original image')
plt.imshow(image,cmap='gray')

plt.subplot(2,1,2)
plt.title('Enhanced image')
plt.imshow(enhanced_image,cmap='gray')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()



