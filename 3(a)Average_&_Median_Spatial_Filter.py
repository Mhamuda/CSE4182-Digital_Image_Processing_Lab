import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from  math import log10,sqrt

def PSNR(original_img, filtered_img):
    # mse = np.mean((original_img-filtered_img)**2)
    mse = np.mean(np.square(np.subtract(original_img.astype(np.int16),filtered_img.astype(np.int16))))
    if(mse==0):
        return np.Inf
    # original_img = np.array(original_img,dtype=np.float64)
    # filtered_img = np.array(filtered_img,dtype=np.float64)
    max_pixel = 255.0
    # mse = np.mean((original_img-filtered_img)**2)

    # psnr = 20*log10(max_pixel/sqrt(mse))
    psnr = 20 * log10(max_pixel) - 10 * log10(mse)
    psnr = round(psnr,2)
    return psnr

img_path = './lenna.webp'
gray_img = cv2.imread(img_path,0)
image = cv2.resize(gray_img,(512,512))

row,col = image.shape
image_size = row
# print(image_size)

plt.figure(figsize=(7,7))
plt.subplot(2,2,1)
plt.title('Original image')
plt.imshow(image,cmap='gray')

# ----------Adding salt and pepper noise-----------
noisy_img = image.copy()
l = int(input("Enter the value of l : "))
r = int(input("Enter the value of r : "))
number_of_pixels = random.randint(l,r)
print(number_of_pixels)

# adding salt
for i in range(number_of_pixels):
    x_crd = random.randint(0,col-1)
    y_crd = random.randint(0,row-1)
    noisy_img[y_crd, x_crd] = 255

# adding pepper
for i in range(number_of_pixels):
    x_crd = random.randint(0,col-1)
    y_crd = random.randint(0,row-1)
    noisy_img[y_crd, x_crd] = 0

plt.subplot(2,2,2)
plt.title('Noisy Image')
plt.imshow(noisy_img,cmap='gray')

# --------------Average filter----------------
mask_size = input('Enter the mask size: ')
m = int(mask_size)
n = int(mask_size)
max_size = m*n
mid = m//2
# print(mid)

a = 1/max_size
avg_filter = [[a]*n]*m
avg_image = noisy_img.copy()

# for i in range (m):
#     for j in range(n):
#         print(avg_filter[i][j],end=" ")
#     print()

for i in range(0,image_size,1):
    for j in range(0,image_size,1):
        temp = 0
        for x,c in zip (range(i-mid,i+mid+1,1),range(0,m,1)):
            for y,d in zip (range(j-mid,j+mid+1,1),range(0,n,1)):
                if(x>=0 and x<image_size and y>=0 and y<image_size):
                    temp+=(noisy_img[x][y]*avg_filter[c][d])
        avg_image[i,j]=temp


plt.subplot(2,2,3)
plt.title('Average filtered image')
plt.imshow(avg_image,cmap='gray')

# avg2_image = cv2.blur(image, (m,n))
# plt.subplot(2,2,2)
# plt.title('Average filtered image (function)')
# plt.imshow(avg2_image,cmap='gray')

# --------------Median Filter--------------
median_image = noisy_img.copy()
median_mid = max_size//2
# print(median_mid)

for i in range(image_size):
    for j in range(image_size):
        temp = []
        for x in range(i-mid,i+mid+1,1):
            for y in range(j-mid,j+mid+1,1):
                if(0<=x<image_size  and 0<=y<image_size):
                    temp.append(noisy_img[x, y])
                else:
                    temp.append(0)
        
        temp = sorted(temp)
        median_image[i, j] = (temp[median_mid])

plt.subplot(2,2,4)
plt.title('Median Filtered Image')
plt.imshow(median_image,cmap='gray')  

# median2_image = cv2.medianBlur(image, m)
# plt.subplot(2,2,4)
# plt.title('Median Filtered Image (function)')
# plt.imshow(median2_image,cmap='gray')

psnr_noisy = PSNR(image, noisy_img)
psnr_avg = PSNR(image, avg_image)
psnr_median = PSNR(image, median_image)

print("Noisy image PSNR = ",psnr_noisy)
print("Average filtered PSNR = ",psnr_avg)
print("Median filtered PSNR = ",psnr_median)

plt.show()
cv2.destroyAllWindows()