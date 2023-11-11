import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = './Rose.tif'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))

img1 = img.copy()
MSB3_img = img1 & 224
diff_img = cv2.absdiff(img,MSB3_img)

plt.figure(figsize=(7,7))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(1,3,1)
plt.title('Original image')
plt.imshow(img,cmap='gray')

plt.subplot(1,3,2)
plt.title('MSB-3 image')
plt.imshow(MSB3_img,cmap='gray')

plt.subplot(1,3,3)
plt.title('Difference image')
plt.imshow(diff_img,cmap='gray')

plt.show()