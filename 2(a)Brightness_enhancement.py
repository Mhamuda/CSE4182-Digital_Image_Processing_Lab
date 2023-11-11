import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = './lenna.webp'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))

height, width = img.shape
enhanced_img = img.copy()

min_graylevel = int(input('Min gray level: '))
max_graylevel = int(input('Max gray level: '))

for i in range(height):
    for j in range(width):
        pixel_val = img[i][j]

        if min_graylevel<=pixel_val<=max_graylevel:
            enhanced_img[i][j] = min(255,pixel_val+50)

plt.figure(figsize=(7,7))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(1,2,1)
plt.title('Original image')
plt.imshow(img,cmap='gray')

plt.subplot(1,2,2)
plt.title('Enhanced image')
plt.imshow(enhanced_img,cmap='gray')

plt.show()