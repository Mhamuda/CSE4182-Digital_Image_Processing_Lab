import cv2
import matplotlib.pyplot as plt
import numpy as np


def Erosion(img, se):
    height, width=  img.shape
    se_h, se_w = se.shape
    mid = se_h // 2
    eroded_img = np.zeros_like(img)

    for i in range(height):
        for j in range(width):
            fit = True
            for m in range(se_h):
                for n in range(se_w):
                    if se[m][n]==1:
                        x = i-mid+m
                        y = j-mid+n
                        if(x<0 or x>=height or y<0 or y>=width or img[x][y]!=1):
                            fit = False
                            break
                if not fit:
                    break
            if fit:
                eroded_img[i][j]=1
    
    return eroded_img

def Dilation(img, se):
    height, width = img.shape
    se_h, se_w = se.shape
    mid = se_h // 2
    dilated_img = np.zeros_like(img)

    for i in range(height):
        for j in range(width):
            hit = False
            for m in range(se_h):
                for n in range(se_w):
                    if se[m][n]==1:
                        x = i-mid+m
                        y = j-mid+n
                        if(0<=x<height and 0<=y<width and img[x][y]==1):
                            hit = True
                            break
                if hit:
                    break
            if hit:
                dilated_img[i][j]=1
    
    return dilated_img


img_path = './Noisy_Fingerprint.tif'
gray_img = cv2.imread(img_path,0)
img = (gray_img>128).astype(np.uint8)

structuring_element_size = 3
se = np.ones((structuring_element_size,structuring_element_size),dtype=int)

eroded_img = Erosion(img, se)
dilated_img = Dilation(eroded_img, se)

plt.figure(figsize=(7,7))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(1,3,1)
plt.title('Binary image')
plt.imshow(img,cmap='gray')

plt.subplot(1,3,2)
plt.title('After Erosion')
plt.imshow(eroded_img,cmap='gray')

plt.subplot(1,3,3)
plt.title('After Dilation')
plt.imshow(dilated_img,cmap='gray')

plt.show()