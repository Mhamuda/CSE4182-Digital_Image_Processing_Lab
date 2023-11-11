import cv2
import matplotlib.pyplot as plt
import numpy as np


def FFT(img):
    return np.fft.fftshift(np.fft.fft2(img))

def IFFT(img):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(img)))

def Ideal_Lowpass(F, D0):
    M, N = F.shape
    H = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D<=D0:
                H[u, v] = 1
    
    G = H * F
    return IFFT(G)


img_path = './pattern.tif'
gray_img = cv2.imread(img_path,0)
img = cv2.resize(gray_img,(512,512))
height, width = img.shape

#----Adding gaussian noise-----
mean =  0
std_dev = 25
noise = np.random.randn(height, width)*std_dev + mean
noise = noise.astype(np.uint8)
Gnoisy_img = cv2.add(img, noise)

Gnoisy_img_fft = FFT(Gnoisy_img)
Gnoisy_img_magnitude = np.log(np.abs(Gnoisy_img_fft)+1)


plt.figure(figsize=(5,5))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.subplot(3,3,1)
plt.title('Original Image')
plt.imshow(img,cmap='gray')

plt.subplot(3,3,2)
plt.title('Gaussian Noisy Image')
plt.imshow(Gnoisy_img,cmap='gray')

plt.subplot(3,3,3)
plt.title('Gaussian Noisy Image FFT')
plt.imshow(Gnoisy_img_magnitude,cmap='gray')

cutoff_frequencies = [5, 10, 15, 20, 25, 30]
ln = len(cutoff_frequencies)

for i in range(ln):
    D0 = cutoff_frequencies[i]
    filtered_img = Ideal_Lowpass(Gnoisy_img_fft, D0)

    plt.subplot(3,3,i+4)
    plt.title(f'Radious={D0}')
    plt.imshow(filtered_img, cmap='gray')

plt.show()