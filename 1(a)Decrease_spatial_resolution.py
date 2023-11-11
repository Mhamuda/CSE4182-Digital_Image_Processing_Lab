import cv2 
import matplotlib.pyplot as plt
import numpy as np

img_path = './Rose.tif'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))

height, width = img.shape
a = height
i=1

while a>=4:
    img = cv2.resize(img,(a,a))
    plt.subplot(2,4,i)
    plt.title(f'{a} X {a}')
    plt.imshow(img,cmap='gray')
    a//=2
    i+=1

plt.show()