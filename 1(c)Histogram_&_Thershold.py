import cv2 
import matplotlib.pyplot as plt
import numpy as np

img_path = './lenna.webp'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))

th_img = img.copy()
threshold_val = int(input('Threshold value: '))

img_his = [0]*256
th_img_his = [0]*256
height, width = img.shape

for i in range(height):
    for j in range(width):
        pixel_val = img[i][j]
        img_his[pixel_val]+=1

        if pixel_val>threshold_val:
            th_img_his[255]+=1
            th_img[i][j]=255
        else:
            th_img_his[0]+=1
            th_img[i][j]=0

plt.figure(figsize=(5,5))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(2,2,1)
plt.title('Original image')
plt.imshow(img,cmap='gray')

plt.subplot(2,2,2)
plt.bar(range(256),img_his,width=1,color='cyan')
plt.title('Original image histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')

plt.subplot(2,2,3)
plt.title('Thresholdl image')
plt.imshow(th_img,cmap='gray')

plt.subplot(2,2,4)
plt.bar(range(256),th_img_his,width=1,color='red')
plt.title('Threshold image histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')

plt.show()