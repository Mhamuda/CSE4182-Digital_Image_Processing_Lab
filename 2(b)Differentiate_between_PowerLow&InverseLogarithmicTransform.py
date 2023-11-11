import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = './Aerial Image.tif'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))

power_law = img.copy()
inverse_log = img.copy()
gamma = 0.5

power_law = np.power(power_law/255.0,gamma)*255.0
power_law = np.uint8(power_law)

c = 255.0/np.log(1+255.0)
inverse_log = np.exp(inverse_log/c)-1
inverse_log = np.uint8(inverse_log)

plt.figure(figsize=(7,7))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(1,3,1)
plt.title('Original image')
plt.imshow(img,cmap='gray')

plt.subplot(1,3,2)
plt.title('Power law image')
plt.imshow(power_law,cmap='gray')

plt.subplot(1,3,3)
plt.title('Inverse log image')
plt.imshow(inverse_log,cmap='gray')

plt.show()