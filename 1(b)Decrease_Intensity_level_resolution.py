import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = './TestImage.png'
gray_image = cv2.imread(image_path,0)
image = cv2.resize(gray_image,(512,512))

# window_name = "Image"
# window_size = (512,512)
# cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name,window_size)

num_bits = 0
plt.figure(figsize=(3,3))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

# Iterate and decrease intensity level resolution by one bit at a time
for i in range(8):
    image = ((image>>i)<<i) #forst right shift for reducing intensity level then The purpose of this left-shift is to restore the original bit depth of the image
    #cv2.imshow(window_name,image)
    plt.subplot(3,3,i+1)
    plt.title(f'{8-i} bits per pixel')
    plt.imshow(image,cmap='gray')
    num_bits += 1
    cv2.waitKey(1000)
    
plt.show()
cv2.destroyAllWindows()