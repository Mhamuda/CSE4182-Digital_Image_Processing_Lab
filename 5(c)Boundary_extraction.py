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


img_path = './Human_shadow.tif'
gray_img = cv2.imread(img_path,0)
binary_img = (gray_img>128).astype(np.uint8)
img = binary_img.copy()

structuring_element_size = 3
se = np.ones((structuring_element_size,structuring_element_size),dtype=int)

eroded_img = Erosion(img, se)
boundary_extracted_img = binary_img - eroded_img

plt.figure(figsize=(7,7))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(1,2,1)
plt.title('Binary image')
plt.imshow(img,cmap='gray')

plt.subplot(1,2,2)
plt.title('Boundary extracted image')
plt.imshow(boundary_extracted_img,cmap='gray')

plt.show()