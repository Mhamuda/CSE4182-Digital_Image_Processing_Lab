import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def Average_Filter(noisy_img,avg_mask):
    img = noisy_img.copy()
    height, width = img.shape
    ms_h = len(avg_mask)
    ms_w = len(avg_mask[0])
    mid = ms_h // 2

    for i in range(height):
        for j in range(width):
            temp=0
            for m in range(ms_h):
                for n in range(ms_w):
                    x = i-mid+m
                    y = j-mid+n
                    if(0<=x<height and 0<=y<width):
                        temp+=(noisy_img[x][y]*avg_mask[m][n])
            img[i, j] = temp
    
    return img

def Median_Filter(noisy_img, mask_size):
    img = noisy_img.copy()
    height, width = img.shape
    mid = mask_size // 2
    median_mid = (mask_size*mask_size)//2

    for i in range(height):
        for j in range(width):
            temp=[]
            for m in range(mask_size):
                for n in range(mask_size):
                    x = i-mid+m
                    y = j-mid+n
                    if(0<=x<height and 0<=y<width):
                        temp.append(noisy_img[x][y])
                    else:
                        temp.append(0)
            temp=sorted(temp)
            img[i, j] = temp[median_mid]
    
    return img


def PSNR(org_img, filtered_img):
    org_img, filtered_img = np.float64(org_img), np.float64(filtered_img)
    mse = np.mean((org_img-filtered_img)**2)
    if mse == 0:
        return float('inf')
    
    psnr = 20*np.log10(255.0)-10*np.log10(mse)
    psnr = round(psnr,2)
    return psnr


img_path = './lenna.webp'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))

noisy_img = img.copy()

row, col = img.shape
l = int(input('l : '))
r = int(input('r : '))
num_of_pixels = random.randint(l, r)
print(num_of_pixels)

# adding salt
for i in range(num_of_pixels):
    x = random.randint(0, col-1)
    y = random.randint(0, row-1)
    noisy_img[y, x] = 255

# adding pepper
for i in range(num_of_pixels):
    x = random.randint(0, col-1)
    y = random.randint(0, row-1)
    noisy_img[y, x] =  0


mask_size = int(input('Mask size : '))
m = mask_size
n = mask_size
max_size = m*n
a  = 1/max_size

avg_mask = [[a]*n]*m
avg_img = Average_Filter(noisy_img, avg_mask)
median_img = Median_Filter(noisy_img, mask_size)

noisy_psnr = PSNR(img, noisy_img)
avg_psnr = PSNR(img, avg_img)
median_psnr = PSNR(img, median_img)

plt.figure(figsize=(7,7))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(2,2,1)
plt.title('Original image')
plt.imshow(img,cmap='gray')

plt.subplot(2,2,2)
plt.title(f'Noisy image PSNR={noisy_psnr}')
plt.imshow(noisy_img,cmap='gray')

plt.subplot(2,2,3)
plt.title(f'Average filtered image PSNR={avg_psnr}')
plt.imshow(avg_img,cmap='gray')

plt.subplot(2,2,4)
plt.title(f'Median filtered image PSNR={median_psnr}')
plt.imshow(median_img,cmap='gray')

plt.show()