import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from math import log10,sqrt

def average_filter(img, image_size, m, n):
    filtered_img = img.copy()
    mid = m//2
    a = 1/(m*n)
    avg_mask = [[a]*n]*m

    for i in range(0,image_size,1):
        for j in range(image_size):
            temp = 0
            for x,c in zip(range(i-mid,i+mid+1,1), range(0,m,1)):
                for y,d in zip(range(j-mid,j+mid+1,1), range(0,n,1)):
                    if(0<=x<image_size and 0<=y<image_size):
                        temp+=(img[x][y]*avg_mask[c][d])
            filtered_img[i,j]=temp
    
    return filtered_img

def PSNR(original_img, filtered_img):
    mse = np.mean(np.square(np.subtract(original_img.astype(np.int16),filtered_img.astype(np.int16))))
    if mse==0:
        return np.Inf
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel) - 10 * log10(mse)
    psnr = round(psnr,2) 
    return psnr

image_path = './lenna.webp'
gray_image = cv2.imread(image_path,0)
image = cv2.resize(gray_image,(512,512))
row,col = image.shape
image_size = row

noisy_img = image.copy()
l = int(input("Enter the value of l : "))
r = int(input("Enter the value of r : "))
number_of_pixels = random.randint(l,r)

# adding salt
for i in range(number_of_pixels):
    x_crd = random.randint(0,col-1)
    y_crd = random.randint(0,row-1)
    noisy_img[x_crd, y_crd] = 255

# adding pepper
for i in range(number_of_pixels):
    x_crd = random.randint(0,col-1)
    y_crd = random.randint(0,row-1)
    noisy_img[x_crd, y_crd] = 0

mask_3 = average_filter(noisy_img, image_size, 3, 3)
mask_5 = average_filter(noisy_img, image_size, 5, 5)
mask_7 = average_filter(noisy_img, image_size, 7, 7)

psnr_noisy = PSNR(image,noisy_img)
psnr_3 = PSNR(image,mask_3)
psnr_5 = PSNR(image,mask_5)
psnr_7 = PSNR(image,mask_7)

plt.figure(figsize=(7,7))

plt.subplot(2,3,1)
plt.title('Original image')
plt.imshow(image,cmap='gray')

plt.subplot(2,3,2)
plt.title(f'Noisy PSNR {psnr_noisy}')
plt.imshow(noisy_img,cmap='gray')

plt.subplot(2,3,3)
plt.title(f'3x3 PSNR {psnr_3}')
plt.imshow(mask_3,cmap='gray')

plt.subplot(2,3,4)
plt.title(f'5x5 PSNR {psnr_5}')
plt.imshow(mask_5,cmap='gray')

plt.subplot(2,3,5)
plt.title(f'7x7 PSNR {psnr_7}')
plt.imshow(mask_7,cmap='gray')

plt.show()
cv2.destroyAllWindows()