import cv2 
import matplotlib.pyplot as plt
import numpy as np

img_path = './skull.tif'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))


plt.figure(figsize=(3,3))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

for i in range (8):
    img = ((img>>i)<<i)
    plt.subplot(3,3,i+1)
    plt.title(f'{8-i} bits per pixel')
    plt.imshow(img,cmap='gray')

plt.show()