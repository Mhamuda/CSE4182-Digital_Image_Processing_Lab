import cv2
import matplotlib.pyplot as plt

image_path = './pikachu.png'

# cv2.imread(filename, flag)
# filename: The path to the image file.
# flag: The flag specifies the way how the image should be read. (0=gray,1=rgb)
image = cv2.imread(image_path,0)
image2 = cv2.resize(image,(512,512))

# cv2.imshow("Original Image",image2)
window_name = "Image"
window_size = (512,512)
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)  #Setting window name
# cv2.resizeWindow(window_name, width, height)
cv2.resizeWindow(window_name,window_size)       #Fixing window size

# Get the dimensions (width and height) of the image
height, width= image2.shape
a=height

# print(f"Image Width: {width} pixels")
# print(f"Image Height: {height} pixels")

while a>0:
    image2 = cv2.resize(image2,(a,a))   #Perforimg image resize operation
    # cv2.imshow(window_name, image)
    cv2.imshow(window_name,image2)  #Showing the image
    cv2.waitKey(1000)   #Pause for 1 second
    print(a)
    a//=2   #Decrease spatial resolution by half
    


# cv2.destroyAllWindows()
