import cv2
import matplotlib.pyplot as plt

image_path = './pikachu.png'
gray_image = cv2.imread(image_path,0)
image = cv2.resize(gray_image,(512,512))

# window_name = "Image"
# window_size = (512,512)
# cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name,window_size)

thresholdImage = image.copy()
height,width = image.shape
histogram = [0]*256
threshold = [0]*256

threshold_value = int(input('Threshold value : '))

for i in range(0,height):
    for j in range(0,width):
        histogram[image[i][j]]+=1

        if image[i][j]>threshold_value:
            threshold[255]+=1
            thresholdImage[i][j]=255
        else:
            threshold[0]+=1
            thresholdImage[i][j]=0


plt.figure(figsize=(5,5))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
plt.subplot(2,2,1)
plt.title('Original Image')
plt.imshow(image,cmap='gray')

plt.subplot(2,2,2)
plt.bar(range(256),histogram,width=1,color='blue')
# plt.plot(histogram, marker='', linestyle='-', color='b', markersize=8)  # 'o' for markers, '-' for lines, 'b' for blue color
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2,2,3)
plt.title('Single Threshold Image')
plt.imshow(thresholdImage,cmap='gray')

plt.subplot(2,2,4)
plt.bar(range(256),threshold,width=1,color='red')
plt.title('Single Threshold')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
